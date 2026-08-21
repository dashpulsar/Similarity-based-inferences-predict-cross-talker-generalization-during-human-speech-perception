args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: fit_ceiling_compatibility.R MODEL_INPUT.csv OUTPUT_DIR")
}

suppressPackageStartupMessages(library(lme4))

input_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- args[[2L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
input <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("dataset_id", "fold", "ceiling_z", "numCorrect", "numIncorrect")
missing <- setdiff(required, names(input))
if (length(missing) > 0L) stop(paste("missing columns:", paste(missing, collapse = ", ")))
dataset_id <- unique(input$dataset_id)
if (length(dataset_id) != 1L || !(dataset_id %in% c("AN19", "X21", "B23"))) {
  stop("one registered dataset is required")
}
if (!identical(sort(unique(as.integer(input$fold))), 0:2)) stop("expected folds 0, 1, 2")

if (dataset_id == "AN19") {
  dataset_required <- c("Keyword", "TestTalker", "SubjectID")
  formula <- cbind(numCorrect, numIncorrect) ~ ceiling_z + (1 + ceiling_z | SubjectID)
} else if (dataset_id == "X21") {
  dataset_required <- c("Keyword", "TestTalkerID", "SentenceID")
  formula <- cbind(numCorrect, numIncorrect) ~ ceiling_z +
    (1 | SentenceID / Keyword) + TestTalkerID
} else {
  dataset_required <- c("Sentence", "TestTalker", "Condition")
  formula <- cbind(numCorrect, numIncorrect) ~ ceiling_z +
    (1 | Sentence) + TestTalker
}
missing <- setdiff(dataset_required, names(input))
if (length(missing) > 0L) stop(paste("missing dataset columns:", paste(missing, collapse = ", ")))

results <- list()
diagnostics <- list()
for (fold_id in 0:2) {
  data <- input[input$fold == fold_id, , drop = FALSE]
  warnings <- character()
  started <- proc.time()[[3L]]
  fitted <- tryCatch(
    withCallingHandlers(
      glmer(
        formula, data = data, family = binomial(link = "logit"),
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1000000L))
      ),
      warning = function(w) {
        warnings <<- c(warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )
  elapsed <- proc.time()[[3L]] - started
  if (inherits(fitted, "error")) stop(sprintf("fold %d failed: %s", fold_id, conditionMessage(fitted)))
  table <- coef(summary(fitted))
  if (!("ceiling_z" %in% rownames(table))) stop("ceiling coefficient missing")
  value <- table["ceiling_z", ]
  messages <- fitted@optinfo$conv$lme4$messages
  message_text <- if (is.null(messages)) "" else paste(messages, collapse = " | ")
  optimizer_code <- fitted@optinfo$conv$opt
  if (is.null(optimizer_code)) optimizer_code <- 0L
  results[[fold_id + 1L]] <- data.frame(
    dataset_id = dataset_id, feature_key = "behavioral_ceiling", fold = fold_id,
    scope = "heldout_refit", publication_status = "notebook_compatibility",
    predictor = "training_item_logodds", estimate = unname(value[["Estimate"]]),
    std_error = unname(value[["Std. Error"]]), z_value = unname(value[["z value"]]),
    p_value = unname(value[["Pr(>|z|)"]]),
    conf_low = unname(value[["Estimate"]] - 1.96 * value[["Std. Error"]]),
    conf_high = unname(value[["Estimate"]] + 1.96 * value[["Std. Error"]]),
    stringsAsFactors = FALSE
  )
  diagnostics[[fold_id + 1L]] <- data.frame(
    dataset_id = dataset_id, fold = fold_id, n_rows = nrow(data),
    formula = paste(deparse(formula), collapse = " "), fit_ok = TRUE,
    converged = identical(as.integer(optimizer_code), 0L) && identical(message_text, ""),
    singular = isSingular(fitted, tol = 1e-4),
    optimizer_code = paste(optimizer_code, collapse = ";"),
    convergence_messages = message_text,
    warnings = paste(unique(warnings), collapse = " | "),
    elapsed_seconds = elapsed, stringsAsFactors = FALSE
  )
}

write.csv(do.call(rbind, results), file.path(output_dir, "ceiling_coefficients.csv"), row.names = FALSE, na = "")
write.csv(do.call(rbind, diagnostics), file.path(output_dir, "diagnostics.csv"), row.names = FALSE, na = "")
write.csv(
  data.frame(
    R_version = R.version.string, lme4_version = as.character(packageVersion("lme4")),
    Matrix_version = as.character(packageVersion("Matrix")), input_path = input_path,
    dataset_id = dataset_id, interpretation = "heldout refit; association stability, not prediction",
    stringsAsFactors = FALSE
  ),
  file.path(output_dir, "software.csv"), row.names = FALSE
)
