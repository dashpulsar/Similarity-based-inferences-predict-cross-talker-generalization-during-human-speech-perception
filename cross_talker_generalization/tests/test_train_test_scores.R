# Focused scientific checks for matched train/test predictive log loss.
# Run from the repository root:
# Rscript cross_talker_generalization/tests/test_train_test_scores.R
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
script <- if (length(args)) args[[1L]] else "cross_talker_generalization/R/fit_confirmatory.R"
script <- normalizePath(script, mustWork = TRUE)
scratch <- tempfile("matched-train-test-")
dir.create(scratch)

set.seed(20260917)
d <- expand.grid(participant = seq_len(36L), item = seq_len(12L))
d$dataset_id <- "synthetic"
d$feature_key <- "tr_24"
d$participant_id <- sprintf("p%02d", d$participant)
d$fold <- (d$participant - 1L) %% 3L
d$condition_id <- ifelse(d$participant %% 2L, "control", "exposure")
d$analysis_item_id <- sprintf("word%02d", d$item)
d$test_talker_id <- ifelse(d$item %% 2L, "talker1", "talker2")
d$raw_distance <- rnorm(nrow(d), mean = 0.4 * d$fold, sd = 1)
d$predictor_status <- "available"
participant_effect <- rnorm(36L, sd = 0.6)
item_effect <- rnorm(12L, sd = 0.5)
eta <- 0.5 - 0.7 * d$raw_distance + 0.4 * (d$condition_id == "exposure") +
  participant_effect[d$participant] + item_effect[d$item]
trials <- sample(c(1L, 3L, 5L, 10L), nrow(d), replace = TRUE)
d$response_correct <- rbinom(nrow(d), size = trials, prob = plogis(eta))
d$response_incorrect <- trials - d$response_correct
input_path <- file.path(scratch, "input.csv")
output_path <- file.path(scratch, "scores")
write.csv(d, input_path, row.names = FALSE)

# Observe the actual fits while running the production script.  Full-data fits
# remain descriptive outputs; every CV fit must contain exactly two folds.
fit_fold_sets <- list()
run_env <- new.env(parent = globalenv())
run_env$commandArgs <- function(trailingOnly = FALSE) {
  c(input_path, output_path, "raw_distance", "-1", "similarity_z", "all")
}
run_env$glmer <- function(formula, data, ...) {
  fit_fold_sets[[length(fit_fold_sets) + 1L]] <<- sort(unique(data$fold))
  lme4::glmer(formula, data = data, ...)
}
sys.source(script, envir = run_env)
scores <- read.csv(file.path(output_path, "train_test_scores.csv"))
metrics <- read.csv(file.path(output_path, "cv_metrics.csv"))
predictions <- read.csv(file.path(output_path, "oof_predictions.csv"))
near <- function(a, b) isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = 1e-10))

stopifnot(nrow(scores) == 3L * 4L * 2L,
          !anyDuplicated(scores[c("feature_key", "fold", "model_id", "split")]),
          all(scores$score_status == "ok"),
          all(scores$fit_ok),
          all(scores$prediction_convention == "fixed_effects_only_re_form_NA"),
          all(scores$likelihood_convention == "word_response_bernoulli_no_binomial_coefficient"),
          near(scores$mean_log_likelihood, -scores$mean_log_loss),
          near(scores$total_log_likelihood, -scores$total_log_loss),
          near(scores$mean_log_loss, scores$total_log_loss / scores$total_trials),
          all(scores$total_trials > scores$n_rows))
stopifnot(all(vapply(fit_fold_sets, function(f) length(f) %in% c(2L, 3L), logical(1L))))
for (fold in 0:2) {
  stopifnot(any(vapply(fit_fold_sets, function(f) identical(f, setdiff(0:2, fold)), logical(1L))))
  rows <- scores[scores$fold == fold, ]
  train <- d[d$fold != fold, ]
  test <- d[d$fold == fold, ]
  stopifnot(near(rows$train_predictor_mean, rep(mean(train$raw_distance), nrow(rows))),
            near(rows$train_predictor_sd, rep(sd(train$raw_distance), nrow(rows))))
  for (split in c("train", "test")) {
    split_d <- if (split == "train") train else test
    split_rows <- rows[rows$split == split, ]
    stopifnot(all(split_rows$n_rows == nrow(split_d)),
              all(split_rows$total_trials == sum(split_d$response_correct + split_d$response_incorrect)))
  }
  for (model in unique(rows$model_id)) {
    s <- rows[rows$model_id == model & rows$split == "test", ]
    m <- metrics[metrics$scope == "oof_fold" & metrics$fold == fold & metrics$model_id == model, ]
    p <- predictions[predictions$fold == fold & predictions$model_id == model, ]
    stopifnot(nrow(m) == 1L,
              near(s$total_log_loss, m$total_log_loss),
              near(s$mean_log_loss, m$mean_log_loss),
              near(s$total_log_loss, sum(p$log_loss)))
  }
}

# Reconstruct the last training fit's two split scores directly, using the same
# frozen training coefficients and fixed-effect prediction convention.
last_fold <- 2L
for (model in names(run_env$train_scope$bundles)) {
  fit <- run_env$train_scope$bundles[[model]]$fit
  for (split in c("train", "test")) {
    split_d <- if (split == "train") d[d$fold != last_fold, ] else d[d$fold == last_fold, ]
    train_d <- d[d$fold != last_fold, ]
    split_d$similarity_z <- -(split_d$raw_distance - mean(train_d$raw_distance)) / sd(train_d$raw_distance)
    p <- predict(fit, newdata = split_d, type = "response", re.form = NA, allow.new.levels = TRUE)
    p <- pmin(pmax(p, 1e-12), 1 - 1e-12)
    expected <- -sum(split_d$response_correct * log(p) + split_d$response_incorrect * log1p(-p))
    row <- scores[scores$fold == last_fold & scores$model_id == model & scores$split == split, ]
    stopifnot(near(row$total_log_loss, expected))
  }
}

# Verify grouping invariance: one grouped count gives the same loss as the
# corresponding individual responses.  Failed fits stay explicit, never zero.
grouped <- run_env$binomial_loss(3, 2, 0.7)
ungrouped <- sum(run_env$binomial_loss(c(1, 1, 1, 0, 0), c(0, 0, 0, 1, 1), rep(0.7, 5)))
failed <- run_env$score_split(list(fit = NULL), d)
stopifnot(near(grouped, ungrouped), failed$status == "fit_failed", all(is.na(failed$loss)))
stopifnot(identical(names(metrics), c("dataset_id", "feature_key", "scope", "fold", "model_id",
                                   "n_rows", "total_trials", "total_log_loss", "mean_log_loss")))
cat("PASS: 24 matched split scores; frozen training scaling/model; response-weighted loss;\n")
cat("      identical held-out metrics; direct fixed-effect scoring; grouped-count invariance.\n")
