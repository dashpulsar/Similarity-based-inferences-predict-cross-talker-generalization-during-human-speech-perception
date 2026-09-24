args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 3L)
suppressPackageStartupMessages(library(lme4))
d <- read.csv(args[1], stringsAsFactors=FALSE)
k <- as.numeric(args[3])
stopifnot(all(d$fold != 0), all(is.finite(d$raw_distance)), k > 0)
for (col in c("participant_id", "analysis_item_id", "test_talker_id")) d[[col]] <- factor(d[[col]])
s <- exp(-k * d$raw_distance)
stopifnot(is.finite(sd(s)), sd(s) > 0)
d$similarity_z <- (s - mean(s))/sd(s)
warnings <- character()
t0 <- proc.time()[[3]]
fit <- withCallingHandlers(tryCatch(glmer(
    cbind(response_correct, response_incorrect) ~ similarity_z +
    (1 | participant_id) + (1 | analysis_item_id) + (1 | test_talker_id),
    data=d, family=binomial(link="logit"), nAGQ=1L,
    control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=20000), calc.derivs=TRUE)
), error=function(e) {warnings <<- c(warnings, conditionMessage(e)); NULL}),
warning=function(w) {warnings <<- c(warnings, conditionMessage(w)); invokeRestart("muffleWarning")})
result <- data.frame(status="failed", z=NA_real_, log_likelihood=NA_real_,
    coefficient=NA_real_, std_error=NA_real_, singular=NA, convergence="failed",
    n_rows=nrow(d), participants=nlevels(d$participant_id),
    n_word_responses=sum(d$response_correct+d$response_incorrect),
    similarity_mean=mean(s), similarity_sd=sd(s), fit_seconds=proc.time()[[3]]-t0,
    warnings=paste(unique(warnings), collapse=" | "), lme4_version=as.character(packageVersion("lme4")))
if (!is.null(fit)) {
    c <- coef(summary(fit))["similarity_z",]
    messages <- fit@optinfo$conv$lme4$messages
    optimizer_code <- fit@optinfo$conv$opt
    ok <- (is.null(optimizer_code) || all(optimizer_code==0))
    # A boundary/singular estimate is retained, flagged, and uses the same model structure.
    critical <- if (is.null(messages)) character() else messages[!grepl("boundary.*singular", messages)]
    result$status <- if (ok && length(critical)==0 && all(is.finite(c))) "ok" else "nonconverged"
    result$convergence <- if (is.null(messages)) "ok" else paste(messages,collapse=" | ")
    result$z <- c["z value"]
    result$coefficient <- c["Estimate"]
    result$std_error <- c["Std. Error"]
    result$log_likelihood <- as.numeric(logLik(fit))
    result$singular <- isSingular(fit, tol=1e-4)
}
write.csv(result, args[2], row.names=FALSE)
