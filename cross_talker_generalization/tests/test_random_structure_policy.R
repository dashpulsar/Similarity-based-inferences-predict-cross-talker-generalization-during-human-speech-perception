# Scientific check: optional declared participant/item structure is fixed, while
# omitting the option and explicitly asking for registered give identical fits.
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = TRUE)
script <- if (length(args)) args[[1L]] else "cross_talker_generalization/R/fit_confirmatory.R"
script <- normalizePath(script, mustWork = TRUE)
scratch <- tempfile("random-policy-")
dir.create(scratch)
set.seed(20260917)
d <- expand.grid(participant = 1:30, item = 1:15)
d$dataset_id <- "synthetic"
d$feature_key <- "example"
d$participant_id <- paste0("p", d$participant)
d$fold <- (d$participant - 1L) %% 3L
d$condition_id <- ifelse(d$participant %% 2L, "A", "B")
d$analysis_item_id <- paste0("w", d$item)
d$test_talker_id <- paste0("t", (d$participant + d$item) %% 6L)
d$raw_distance <- rnorm(nrow(d))
d$predictor_status <- "available"
eta <- 0.3 - 0.6 * d$raw_distance + rnorm(30, sd = .6)[d$participant] + rnorm(15, sd = .5)[d$item]
d$response_correct <- rbinom(nrow(d), 5, plogis(eta))
d$response_incorrect <- 5 - d$response_correct
input <- file.path(scratch, "input.csv")
write.csv(d, input, row.names = FALSE)
run <- function(policy = NULL) {
  directory <- file.path(scratch, if (is.null(policy)) "default" else policy)
  env <- new.env(parent = globalenv())
  env$commandArgs <- function(trailingOnly = FALSE) c(input, directory, "raw_distance", "-1", "similarity_z", "predictor_only", policy)
  sys.source(script, envir = env)
  list(env = env, directory = directory)
}
default <- run()
registered <- run("registered")
fixed <- run("participant_item")
for (name in c("coefficients.csv", "cv_metrics.csv", "diagnostics.csv", "variance_components.csv", "train_test_scores.csv")) {
  stopifnot(isTRUE(all.equal(read.csv(file.path(default$directory, name)),
                            read.csv(file.path(registered$directory, name)))))
}
diagnostics <- read.csv(file.path(fixed$directory, "diagnostics.csv"))
components <- read.csv(file.path(fixed$directory, "variance_components.csv"))
stopifnot(all(diagnostics$random_policy == "participant_item"),
          all(diagnostics$random_structure == "participant_item_intercepts"),
          !any(grepl("test_talker", diagnostics$formula)),
          all(grepl("no automatic fallback", diagnostics$selection_reason)),
          setequal(components$group, c("participant_id", "analysis_item_id")),
          nrow(components) == nrow(diagnostics) * 2L,
          isTRUE(all.equal(components$variance, components$stddev_or_correlation^2)),
          all(components$variance_below_1e_8 == (components$variance < 1e-8)))
# A boundary estimate must be retained. A fake singular result verifies that
# this policy does not invoke any automatic deletion of random-effect terms.
calls <- 0L
fixed$env$fit_one <- function(model_id, data, formulas) {
  calls <<- calls + 1L
  list(fit = TRUE, singular = TRUE, convergence = "boundary (singular) fit")
}
scope <- fixed$env$fit_scope(d)
stopifnot(calls == 2L, all(vapply(scope$bundles, function(x) x$singular, logical(1))),
          scope$random_structure == "participant_item_intercepts")
cat("PASS: default equals explicit registered; fixed participant/item policy; exported variance components; boundary retained.\n")
