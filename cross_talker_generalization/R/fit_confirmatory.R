args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2L || length(args) > 6L) {
  stop("usage: fit_confirmatory.R MODEL_INPUT.csv OUTPUT_DIR [PREDICTOR_COLUMN] [DIRECTION] [TERM_NAME] [MODEL_SET]")
}

suppressPackageStartupMessages(library(lme4))

input_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- args[[2L]]
predictor_column <- if (length(args) >= 3L) args[[3L]] else "raw_distance"
predictor_direction <- if (length(args) >= 4L) as.numeric(args[[4L]]) else -1.0
predictor_term <- if (length(args) >= 5L) args[[5L]] else "similarity_z"
model_set <- if (length(args) >= 6L) args[[6L]] else "all"
if (!(predictor_direction %in% c(-1, 1))) stop("predictor direction must be -1 or 1")
if (!grepl("^[A-Za-z][A-Za-z0-9_]*$", predictor_term)) stop("invalid predictor term")
if (!(model_set %in% c("all", "predictor_only"))) stop("model set must be all or predictor_only")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE,
                  na.strings = c("", "NA", "NaN"))
required <- c(
  "dataset_id", "feature_key", "participant_id", "fold", "condition_id",
  "analysis_item_id", "test_talker_id", "response_correct",
  "response_incorrect", predictor_column, "predictor_status"
)
missing <- setdiff(required, names(input))
if (length(missing) > 0L) stop(paste("missing columns:", paste(missing, collapse = ", ")))
if (length(unique(input$dataset_id)) != 1L) stop("one dataset per run is required")
if (!all(input$response_correct >= 0L) || !all(input$response_incorrect >= 0L) ||
    !all(input$response_correct + input$response_incorrect > 0L)) {
  stop("invalid binomial counts")
}

input$analysis_row_id <- seq_len(nrow(input))
input$predictor_value_internal <- input[[predictor_column]]
input <- input[input$predictor_status == "available" & is.finite(input$predictor_value_internal), , drop = FALSE]
if (nrow(input) == 0L) stop("no available predictor rows")
fold_ids <- sort(unique(as.integer(input$fold)))
if (!identical(fold_ids, 0:2)) stop("expected participant folds 0, 1, 2")

condition_levels <- sort(unique(input$condition_id))
input$condition_id <- factor(input$condition_id, levels = condition_levels)
input$participant_id <- factor(input$participant_id)
input$analysis_item_id <- factor(input$analysis_item_id)
input$test_talker_id <- factor(input$test_talker_id)

response_text <- "cbind(response_correct, response_incorrect)"
if (length(unique(input$test_talker_id)) <= 4L) {
  # Four talkers do not support a stable variance-component estimate.  Treat
  # talker as a blocking factor for X21/B23 and retain participant/item REs.
  talker_fixed <- "test_talker_id + "
  random_text <- "(1 | participant_id) + (1 | analysis_item_id)"
  talker_strategy <- "fixed_block"
} else {
  talker_fixed <- ""
  random_text <- "(1 | participant_id) + (1 | analysis_item_id) + (1 | test_talker_id)"
  talker_strategy <- "random_intercept"
}
formula_set <- function(random_structure) {
  formulas <- c(
    M_condition = paste0(response_text, " ~ condition_id + ", talker_fixed, random_structure),
    M_predictor = paste0(response_text, " ~ ", predictor_term, " + ", talker_fixed, random_structure),
    M_joint = paste0(response_text, " ~ condition_id + ", predictor_term, " + ", talker_fixed, random_structure)
  )
  if (model_set == "predictor_only") formulas["M_predictor"] else formulas
}
primary_formula_text <- formula_set(random_text)
fallback_random_text <- "(1 | participant_id) + (1 | analysis_item_id)"
fallback_formula_text <- formula_set(fallback_random_text)

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

fit_scope <- function(data) {
  formulas <- primary_formula_text
  bundles <- lapply(names(formulas), function(model_id) fit_one(model_id, data, formulas))
  names(bundles) <- names(formulas)
  unacceptable <- vapply(
    bundles,
    function(bundle) is.null(bundle$fit) || isTRUE(bundle$singular) || bundle$convergence != "ok",
    logical(1L)
  )
  if (talker_strategy == "random_intercept" && any(unacceptable)) {
    formulas <- fallback_formula_text
    bundles <- lapply(names(formulas), function(model_id) fit_one(model_id, data, formulas))
    names(bundles) <- names(formulas)
    return(list(
      bundles = bundles, formulas = formulas,
      random_structure = "participant_item_intercepts",
      selection_reason = "talker random-intercept candidate was singular or failed convergence"
    ))
  }
  list(
    bundles = bundles, formulas = formulas,
    random_structure = if (talker_strategy == "random_intercept") {
      "participant_item_talker_intercepts"
    } else {
      "participant_item_intercepts_with_fixed_talker_block"
    },
    selection_reason = "primary registered structure accepted"
  )
}

coefficient_rows <- list()
diagnostic_rows <- list()
lrt_rows <- list()
prediction_rows <- list()
metric_rows <- list()
coefficient_cursor <- 1L
diagnostic_cursor <- 1L
lrt_cursor <- 1L
prediction_cursor <- 1L
metric_cursor <- 1L

record_fit <- function(bundle, feature_key, scope, fold, model_id, n_rows,
                       formulas, random_structure, selection_reason) {
  diagnostic_rows[[diagnostic_cursor]] <<- data.frame(
    dataset_id = unique(input$dataset_id), feature_key = feature_key,
    scope = scope, fold = fold, model_id = model_id,
    formula = formulas[[model_id]], optimizer = bundle$optimizer,
    random_structure = random_structure,
    selection_reason = selection_reason, n_rows = n_rows,
    fit_ok = !is.null(bundle$fit), singular = bundle$singular,
    convergence = bundle$convergence, warnings = bundle$warnings,
    error = bundle$error,
    log_likelihood = if (is.null(bundle$fit)) NA_real_ else as.numeric(logLik(bundle$fit)),
    deviance = if (is.null(bundle$fit)) NA_real_ else deviance(bundle$fit),
    AIC = if (is.null(bundle$fit)) NA_real_ else AIC(bundle$fit),
    n_observations = if (is.null(bundle$fit)) NA_integer_ else nobs(bundle$fit),
    stringsAsFactors = FALSE
  )
  diagnostic_cursor <<- diagnostic_cursor + 1L
  if (is.null(bundle$fit)) return(invisible(NULL))
  table <- coef(summary(bundle$fit))
  for (term in rownames(table)) {
    coefficient_rows[[coefficient_cursor]] <<- data.frame(
      dataset_id = unique(input$dataset_id), feature_key = feature_key,
      scope = scope, fold = fold, model_id = model_id, term = term,
      estimate = table[term, "Estimate"], std_error = table[term, "Std. Error"],
      z_value = table[term, "z value"], p_value = table[term, "Pr(>|z|)"],
      conf_low = table[term, "Estimate"] - 1.96 * table[term, "Std. Error"],
      conf_high = table[term, "Estimate"] + 1.96 * table[term, "Std. Error"],
      stringsAsFactors = FALSE
    )
    coefficient_cursor <<- coefficient_cursor + 1L
  }
}

binomial_loss <- function(correct, incorrect, probability) {
  clipped <- pmin(pmax(probability, 1e-12), 1 - 1e-12)
  -(correct * log(clipped) + incorrect * log1p(-clipped))
}

feature_keys <- sort(unique(input$feature_key))
for (feature_key in feature_keys) {
  layer_data <- input[input$feature_key == feature_key, , drop = FALSE]
  full_mean <- mean(layer_data$predictor_value_internal)
  full_sd <- sd(layer_data$predictor_value_internal)
  if (!is.finite(full_sd) || full_sd <= 0) stop(paste("zero distance SD", feature_key))
  layer_data[[predictor_term]] <- predictor_direction * (layer_data$predictor_value_internal - full_mean) / full_sd

  full_scope <- fit_scope(layer_data)
  full_bundles <- full_scope$bundles
  for (model_id in names(full_scope$formulas)) {
    record_fit(
      full_bundles[[model_id]], feature_key, "full", NA_integer_, model_id,
      nrow(layer_data), full_scope$formulas, full_scope$random_structure,
      full_scope$selection_reason
    )
  }
  comparison_specs <- if (model_set == "all") {
    list(
      predictor_beyond_condition = c(reduced = "M_condition", full = "M_joint"),
      condition_beyond_predictor = c(reduced = "M_predictor", full = "M_joint")
    )
  } else {
    list()
  }
  for (comparison_id in names(comparison_specs)) {
    reduced_id <- comparison_specs[[comparison_id]][["reduced"]]
    full_id <- comparison_specs[[comparison_id]][["full"]]
    if (is.null(full_bundles[[reduced_id]]$fit) || is.null(full_bundles[[full_id]]$fit)) next
    comparison <- tryCatch(
      anova(full_bundles[[reduced_id]]$fit, full_bundles[[full_id]]$fit, test = "Chisq"),
      error = function(e) e
    )
    if (inherits(comparison, "error")) {
      lrt_rows[[lrt_cursor]] <- data.frame(
        dataset_id = unique(input$dataset_id), feature_key = feature_key,
        comparison_id = comparison_id, reduced_model = reduced_id, full_model = full_id,
        scope = "full", fold = NA_integer_, chisq = NA_real_, df = NA_real_,
        p_value = NA_real_, status = "error", reason = conditionMessage(comparison)
      )
    } else {
      value_at <- function(name, index = 2L) {
        if (!(name %in% names(comparison)) || length(comparison[[name]]) < index) return(NA_real_)
        as.numeric(comparison[[name]][[index]])
      }
      lrt_rows[[lrt_cursor]] <- data.frame(
        dataset_id = unique(input$dataset_id), feature_key = feature_key,
        comparison_id = comparison_id, reduced_model = reduced_id, full_model = full_id,
        scope = "full", fold = NA_integer_, chisq = value_at("Chisq"),
        df = if (is.finite(value_at("Chi Df"))) value_at("Chi Df") else value_at("Df"),
        p_value = value_at("Pr(>Chisq)"),
        status = "ok", reason = NA_character_
      )
    }
    lrt_cursor <- lrt_cursor + 1L
  }

  for (fold_id in fold_ids) {
    train <- layer_data[layer_data$fold != fold_id, , drop = FALSE]
    test <- layer_data[layer_data$fold == fold_id, , drop = FALSE]
    if (!setequal(unique(train$condition_id), condition_levels)) {
      stop(paste("training fold lacks a condition level", feature_key, fold_id))
    }
    train_mean <- mean(train$predictor_value_internal)
    train_sd <- sd(train$predictor_value_internal)
    if (!is.finite(train_sd) || train_sd <= 0) stop("invalid training-fold distance scale")
    train[[predictor_term]] <- predictor_direction * (train$predictor_value_internal - train_mean) / train_sd
    test[[predictor_term]] <- predictor_direction * (test$predictor_value_internal - train_mean) / train_sd

    train_scope <- fit_scope(train)
    for (model_id in names(train_scope$formulas)) {
      bundle <- train_scope$bundles[[model_id]]
      record_fit(
        bundle, feature_key, "cv_train", fold_id, model_id, nrow(train),
        train_scope$formulas, train_scope$random_structure,
        train_scope$selection_reason
      )
      if (is.null(bundle$fit)) next
      probability <- tryCatch(
        predict(bundle$fit, newdata = test, type = "response", re.form = NA,
                allow.new.levels = TRUE),
        error = function(e) e
      )
      if (inherits(probability, "error")) stop(conditionMessage(probability))
      loss <- binomial_loss(test$response_correct, test$response_incorrect, probability)
      prediction_rows[[prediction_cursor]] <- data.frame(
        dataset_id = unique(input$dataset_id), feature_key = feature_key,
        fold = fold_id, model_id = model_id,
        analysis_row_id = test$analysis_row_id,
        participant_id = as.character(test$participant_id),
        analysis_item_id = as.character(test$analysis_item_id),
        condition_id = as.character(test$condition_id),
        response_correct = test$response_correct,
        response_incorrect = test$response_incorrect,
        probability = as.numeric(probability), log_loss = loss,
        stringsAsFactors = FALSE
      )
      prediction_cursor <- prediction_cursor + 1L
      metric_rows[[metric_cursor]] <- data.frame(
        dataset_id = unique(input$dataset_id), feature_key = feature_key,
        scope = "oof_fold", fold = fold_id, model_id = model_id,
        n_rows = nrow(test), total_trials = sum(test$response_correct + test$response_incorrect),
        total_log_loss = sum(loss), mean_log_loss = sum(loss) / sum(test$response_correct + test$response_incorrect),
        stringsAsFactors = FALSE
      )
      metric_cursor <- metric_cursor + 1L
    }
  }
}

coefficients <- if (length(coefficient_rows)) do.call(rbind, coefficient_rows) else data.frame()
diagnostics <- if (length(diagnostic_rows)) do.call(rbind, diagnostic_rows) else data.frame()
lrts <- if (length(lrt_rows)) do.call(rbind, lrt_rows) else data.frame(
  dataset_id = character(), feature_key = character(), comparison_id = character(),
  reduced_model = character(), full_model = character(), scope = character(),
  fold = integer(), chisq = numeric(), df = numeric(), p_value = numeric(),
  status = character(), reason = character()
)
predictions <- if (length(prediction_rows)) do.call(rbind, prediction_rows) else data.frame()
metrics <- if (length(metric_rows)) do.call(rbind, metric_rows) else data.frame()

if (nrow(metrics) > 0L) {
  overall <- aggregate(
    cbind(n_rows, total_trials, total_log_loss) ~ dataset_id + feature_key + model_id,
    data = metrics, FUN = sum
  )
  overall$scope <- "oof_all"
  overall$fold <- NA_integer_
  overall$mean_log_loss <- overall$total_log_loss / overall$total_trials
  overall <- overall[, names(metrics)]
  metrics <- rbind(metrics, overall)
}

write.csv(coefficients, file.path(output_dir, "coefficients.csv"), row.names = FALSE, na = "")
write.csv(diagnostics, file.path(output_dir, "diagnostics.csv"), row.names = FALSE, na = "")
write.csv(lrts, file.path(output_dir, "likelihood_ratio_tests.csv"), row.names = FALSE, na = "")
write.csv(predictions, file.path(output_dir, "oof_predictions.csv"), row.names = FALSE, na = "")
write.csv(metrics, file.path(output_dir, "cv_metrics.csv"), row.names = FALSE, na = "")
write.csv(
  data.frame(
    R_version = R.version.string,
    lme4_version = as.character(packageVersion("lme4")),
    input_path = input_path,
    n_input_rows = nrow(input),
    dataset_id = unique(input$dataset_id),
    talker_strategy = talker_strategy,
    predictor_column = predictor_column,
    predictor_direction = predictor_direction,
    predictor_term = predictor_term,
    model_set = model_set,
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "software.csv"), row.names = FALSE
)
