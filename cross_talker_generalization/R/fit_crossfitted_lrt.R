args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6L || length(args) > 7L) {
  stop(paste(
    "usage: fit_crossfitted_lrt.R MODEL_INPUT.csv OUTPUT_DIR FEATURE_KEY",
    "PREDICTOR_COLUMN DIRECTION TERM_NAME [INCLUDE_FOLD_BLOCK]"
  ))
}

suppressPackageStartupMessages(library(lme4))

input_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- args[[2L]]
selected_feature_key <- args[[3L]]
predictor_column <- args[[4L]]
predictor_direction <- as.numeric(args[[5L]])
predictor_term <- args[[6L]]
include_fold_block <- if (length(args) >= 7L) {
  tolower(args[[7L]]) %in% c("true", "t", "1", "yes")
} else {
  TRUE
}

if (!(predictor_direction %in% c(-1, 1))) stop("predictor direction must be -1 or 1")
if (!grepl("^[A-Za-z][A-Za-z0-9_]*$", predictor_term)) stop("invalid predictor term")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input <- read.csv(
  input_path, stringsAsFactors = FALSE, check.names = FALSE,
  na.strings = c("", "NA", "NaN")
)
required <- c(
  "dataset_id", "feature_key", "participant_id", "fold", "condition_id",
  "analysis_item_id", "test_talker_id", "response_correct",
  "response_incorrect", predictor_column, "predictor_status"
)
missing <- setdiff(required, names(input))
if (length(missing) > 0L) stop(paste("missing columns:", paste(missing, collapse = ", ")))

input$source_row_id <- seq_len(nrow(input))
input <- input[input$feature_key == selected_feature_key, , drop = FALSE]
if (nrow(input) == 0L) stop(paste("feature not found:", selected_feature_key))
if (length(unique(input$dataset_id)) != 1L) stop("one dataset per run is required")
input$predictor_value_raw <- input[[predictor_column]]
input <- input[
  input$predictor_status == "available" & is.finite(input$predictor_value_raw),
  , drop = FALSE
]
if (nrow(input) == 0L) stop("no available predictor rows")
if (!all(input$response_correct >= 0L) || !all(input$response_incorrect >= 0L) ||
    !all(input$response_correct + input$response_incorrect > 0L)) {
  stop("invalid binomial counts")
}

fold_ids <- sort(unique(as.integer(input$fold)))
if (!identical(fold_ids, 0:2)) stop("expected participant folds 0, 1, 2")
participant_folds <- aggregate(fold ~ participant_id, data = input, FUN = function(x) length(unique(x)))
if (any(participant_folds$fold != 1L)) stop("a participant occurs in more than one fold")

# Construct one predictor value for every observation without using that
# observation's participant fold to estimate the predictor scale.  In the
# current fixed-tau/fixed-k analyses, these training-fold moments are the only
# fold-estimated numerical parameters in the theoretical predictor.
crossfitted_rows <- list()
scale_rows <- list()
for (fold_id in fold_ids) {
  train <- input[as.integer(input$fold) != fold_id, , drop = FALSE]
  test <- input[as.integer(input$fold) == fold_id, , drop = FALSE]
  train_mean <- mean(train$predictor_value_raw)
  train_sd <- sd(train$predictor_value_raw)
  if (!is.finite(train_sd) || train_sd <= 0) {
    stop(paste("invalid training-fold predictor scale for fold", fold_id))
  }
  test[[predictor_term]] <- predictor_direction *
    (test$predictor_value_raw - train_mean) / train_sd
  test$scale_training_mean <- train_mean
  test$scale_training_sd <- train_sd
  crossfitted_rows[[length(crossfitted_rows) + 1L]] <- test
  scale_rows[[length(scale_rows) + 1L]] <- data.frame(
    dataset_id = unique(input$dataset_id),
    feature_key = selected_feature_key,
    held_out_fold = fold_id,
    n_training_rows = nrow(train),
    n_held_out_rows = nrow(test),
    training_mean = train_mean,
    training_sd = train_sd,
    stringsAsFactors = FALSE
  )
}
crossfitted <- do.call(rbind, crossfitted_rows)
crossfitted <- crossfitted[order(crossfitted$source_row_id), , drop = FALSE]
if (nrow(crossfitted) != nrow(input)) stop("cross-fitting did not preserve all eligible rows")
if (anyDuplicated(crossfitted$source_row_id)) stop("cross-fitting duplicated rows")

condition_levels <- sort(unique(crossfitted$condition_id))
crossfitted$condition_id <- factor(crossfitted$condition_id, levels = condition_levels)
crossfitted$participant_id <- factor(crossfitted$participant_id)
crossfitted$analysis_item_id <- factor(crossfitted$analysis_item_id)
crossfitted$test_talker_id <- factor(crossfitted$test_talker_id)
crossfitted$outer_fold <- factor(as.integer(crossfitted$fold), levels = fold_ids)

response_text <- "cbind(response_correct, response_incorrect)"
fold_fixed <- if (include_fold_block) "outer_fold + " else ""
if (length(unique(crossfitted$test_talker_id)) <= 4L) {
  # Match the confirmatory analysis: with only four talkers, use a fixed
  # talker block instead of estimating a talker variance component.
  talker_fixed <- "test_talker_id + "
  primary_random_text <- "(1 | participant_id) + (1 | analysis_item_id)"
  fallback_random_text <- primary_random_text
  talker_strategy <- "fixed_block"
} else {
  talker_fixed <- ""
  primary_random_text <- paste(
    "(1 | participant_id) + (1 | analysis_item_id) +",
    "(1 | test_talker_id)"
  )
  fallback_random_text <- "(1 | participant_id) + (1 | analysis_item_id)"
  talker_strategy <- "random_intercept"
}

formula_set <- function(random_structure) {
  c(
    M_condition = paste0(
      response_text, " ~ condition_id + ", fold_fixed, talker_fixed, random_structure
    ),
    M_predictor = paste0(
      response_text, " ~ ", predictor_term, " + ", fold_fixed,
      talker_fixed, random_structure
    ),
    M_joint = paste0(
      response_text, " ~ condition_id + ", predictor_term, " + ",
      fold_fixed, talker_fixed, random_structure
    )
  )
}

fit_one <- function(model_id, data, formulas) {
  last_bundle <- NULL
  for (optimizer_name in c("bobyqa", "Nelder_Mead", "nloptwrap")) {
    warnings <- character()
    error_message <- NA_character_
    optimizer_control <- if (optimizer_name %in% c("bobyqa", "Nelder_Mead")) {
      list(maxfun = 1000000)
    } else {
      list(maxeval = 1000000)
    }
    fitted <- withCallingHandlers(
      tryCatch(
        glmer(
          as.formula(formulas[[model_id]]), data = data,
          family = binomial(link = "logit"),
          control = glmerControl(
            optimizer = optimizer_name,
            optCtrl = optimizer_control,
            calc.derivs = TRUE
          )
        ),
        error = function(e) {
          error_message <<- conditionMessage(e)
          NULL
        }
      ),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    convergence <- if (is.null(fitted)) {
      error_message
    } else {
      messages <- fitted@optinfo$conv$lme4$messages
      if (is.null(messages)) "ok" else paste(messages, collapse = " | ")
    }
    last_bundle <- list(
      fit = fitted,
      optimizer = optimizer_name,
      warnings = paste(unique(warnings), collapse = " | "),
      error = error_message,
      convergence = convergence,
      singular = if (is.null(fitted)) NA else isSingular(fitted, tol = 1e-4)
    )
    if (!is.null(fitted) && convergence == "ok") return(last_bundle)
  }
  last_bundle
}

primary_formulas <- formula_set(primary_random_text)
bundles <- lapply(names(primary_formulas), function(model_id) {
  fit_one(model_id, crossfitted, primary_formulas)
})
names(bundles) <- names(primary_formulas)
unacceptable <- vapply(
  bundles,
  function(bundle) is.null(bundle$fit) || isTRUE(bundle$singular) || bundle$convergence != "ok",
  logical(1L)
)
if (talker_strategy == "random_intercept" && any(unacceptable)) {
  formulas <- formula_set(fallback_random_text)
  bundles <- lapply(names(formulas), function(model_id) fit_one(model_id, crossfitted, formulas))
  names(bundles) <- names(formulas)
  random_structure <- "participant_item_intercepts"
  structure_reason <- "talker random-intercept candidate was singular or failed convergence"
} else {
  formulas <- primary_formulas
  random_structure <- if (talker_strategy == "random_intercept") {
    "participant_item_talker_intercepts"
  } else {
    "participant_item_intercepts_with_fixed_talker_block"
  }
  structure_reason <- "primary registered structure accepted"
}

diagnostic_rows <- list()
coefficient_rows <- list()
for (model_id in names(bundles)) {
  bundle <- bundles[[model_id]]
  diagnostic_rows[[length(diagnostic_rows) + 1L]] <- data.frame(
    dataset_id = unique(input$dataset_id),
    feature_key = selected_feature_key,
    model_id = model_id,
    formula = formulas[[model_id]],
    optimizer = bundle$optimizer,
    random_structure = random_structure,
    structure_reason = structure_reason,
    n_rows = nrow(crossfitted),
    total_trials = sum(crossfitted$response_correct + crossfitted$response_incorrect),
    fit_ok = !is.null(bundle$fit),
    singular = bundle$singular,
    convergence = bundle$convergence,
    warnings = bundle$warnings,
    error = bundle$error,
    log_likelihood = if (is.null(bundle$fit)) NA_real_ else as.numeric(logLik(bundle$fit)),
    deviance = if (is.null(bundle$fit)) NA_real_ else deviance(bundle$fit),
    AIC = if (is.null(bundle$fit)) NA_real_ else AIC(bundle$fit),
    n_observations = if (is.null(bundle$fit)) NA_integer_ else nobs(bundle$fit),
    stringsAsFactors = FALSE
  )
  if (!is.null(bundle$fit)) {
    coefficient_table <- coef(summary(bundle$fit))
    for (term in rownames(coefficient_table)) {
      coefficient_rows[[length(coefficient_rows) + 1L]] <- data.frame(
        dataset_id = unique(input$dataset_id),
        feature_key = selected_feature_key,
        model_id = model_id,
        term = term,
        estimate = coefficient_table[term, "Estimate"],
        std_error = coefficient_table[term, "Std. Error"],
        z_value = coefficient_table[term, "z value"],
        p_value = coefficient_table[term, "Pr(>|z|)"],
        conf_low = coefficient_table[term, "Estimate"] -
          1.96 * coefficient_table[term, "Std. Error"],
        conf_high = coefficient_table[term, "Estimate"] +
          1.96 * coefficient_table[term, "Std. Error"],
        stringsAsFactors = FALSE
      )
    }
  }
}

lrt_rows <- list()
comparison_specs <- list(
  predictor_beyond_condition = c(reduced = "M_condition", full = "M_joint"),
  condition_beyond_predictor = c(reduced = "M_predictor", full = "M_joint")
)
for (comparison_id in names(comparison_specs)) {
  reduced_id <- comparison_specs[[comparison_id]][["reduced"]]
  full_id <- comparison_specs[[comparison_id]][["full"]]
  reduced_fit <- bundles[[reduced_id]]$fit
  full_fit <- bundles[[full_id]]$fit
  if (is.null(reduced_fit) || is.null(full_fit)) {
    lrt_rows[[length(lrt_rows) + 1L]] <- data.frame(
      dataset_id = unique(input$dataset_id), feature_key = selected_feature_key,
      comparison_id = comparison_id, reduced_model = reduced_id, full_model = full_id,
      reduced_log_likelihood = NA_real_, full_log_likelihood = NA_real_,
      delta_log_likelihood = NA_real_, chisq = NA_real_, df = NA_real_,
      p_value = NA_real_, status = "fit_error", reason = "one or both models failed",
      stringsAsFactors = FALSE
    )
    next
  }
  comparison <- tryCatch(
    anova(reduced_fit, full_fit, test = "Chisq"),
    error = function(e) e
  )
  if (inherits(comparison, "error")) {
    lrt_rows[[length(lrt_rows) + 1L]] <- data.frame(
      dataset_id = unique(input$dataset_id), feature_key = selected_feature_key,
      comparison_id = comparison_id, reduced_model = reduced_id, full_model = full_id,
      reduced_log_likelihood = as.numeric(logLik(reduced_fit)),
      full_log_likelihood = as.numeric(logLik(full_fit)),
      delta_log_likelihood = as.numeric(logLik(full_fit) - logLik(reduced_fit)),
      chisq = NA_real_, df = NA_real_, p_value = NA_real_, status = "error",
      reason = conditionMessage(comparison), stringsAsFactors = FALSE
    )
  } else {
    value_at <- function(name, index = 2L) {
      if (!(name %in% names(comparison)) || length(comparison[[name]]) < index) return(NA_real_)
      as.numeric(comparison[[name]][[index]])
    }
    lrt_rows[[length(lrt_rows) + 1L]] <- data.frame(
      dataset_id = unique(input$dataset_id), feature_key = selected_feature_key,
      comparison_id = comparison_id, reduced_model = reduced_id, full_model = full_id,
      reduced_log_likelihood = as.numeric(logLik(reduced_fit)),
      full_log_likelihood = as.numeric(logLik(full_fit)),
      delta_log_likelihood = as.numeric(logLik(full_fit) - logLik(reduced_fit)),
      chisq = value_at("Chisq"),
      # lme4/R versions label the ANOVA difference column inconsistently.
      # The LRT degrees of freedom are exactly the difference in the fitted
      # models' parameter counts, available from the logLik objects.
      df = as.numeric(attr(logLik(full_fit), "df") - attr(logLik(reduced_fit), "df")),
      p_value = value_at("Pr(>Chisq)"), status = "ok", reason = NA_character_,
      stringsAsFactors = FALSE
    )
  }
}

crossfitted_export <- crossfitted[, c(
  "dataset_id", "feature_key", "source_row_id", "participant_id", "fold",
  "condition_id", "analysis_item_id", "test_talker_id", "response_correct",
  "response_incorrect", "predictor_value_raw", predictor_term,
  "scale_training_mean", "scale_training_sd"
)]
names(crossfitted_export)[names(crossfitted_export) == predictor_term] <- "crossfitted_predictor"

write.csv(crossfitted_export, file.path(output_dir, "crossfitted_predictors.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, scale_rows), file.path(output_dir, "fold_scaling.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, coefficient_rows), file.path(output_dir, "coefficients.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, diagnostic_rows), file.path(output_dir, "diagnostics.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, lrt_rows), file.path(output_dir, "likelihood_ratio_tests.csv"), row.names = FALSE, na = "")
write.csv(
  data.frame(
    R_version = R.version.string,
    lme4_version = as.character(packageVersion("lme4")),
    input_path = input_path,
    dataset_id = unique(input$dataset_id),
    feature_key = selected_feature_key,
    predictor_column = predictor_column,
    predictor_direction = predictor_direction,
    predictor_term = predictor_term,
    include_fold_block = include_fold_block,
    talker_strategy = talker_strategy,
    selection_conditional = TRUE,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "software.csv"), row.names = FALSE
)
