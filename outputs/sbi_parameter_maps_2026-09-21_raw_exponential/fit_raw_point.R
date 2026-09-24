args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L)
suppressPackageStartupMessages(library(lme4))
d <- read.csv(args[1],stringsAsFactors=FALSE)
k <- as.numeric(args[2]); destination <- args[3]
stopifnot(k>0,all(d$fold!=0),nrow(d)==10969,all(is.finite(d$raw_distance)))
for (v in c('participant_id','analysis_item_id','test_talker_id')) d[[v]] <- factor(d[[v]])

# The only predictor transformation in this experiment. No shift, expm1 or z-score.
d$similarity_raw <- exp(-k * d$raw_distance)
s <- d$similarity_raw
model_formula <- cbind(response_correct,response_incorrect) ~
  similarity_raw + test_talker_id + (1|participant_id) + (1|analysis_item_id)
t0 <- proc.time()[[3]]
warnings <- character()
result <- data.frame(k=k,kind='grid',status='running',z=NA_real_,
  log_likelihood=NA_real_,coefficient=NA_real_,std_error=NA_real_,singular=NA,
  var_participant=NA_real_,var_item=NA_real_,n_rows=nrow(d),
  participants=nlevels(d$participant_id),n_word_responses=sum(d$response_correct+d$response_incorrect),
  predictor_mean=mean(s),predictor_sd=sd(s),predictor_min=min(s),predictor_max=max(s),
  n_zero=sum(s==0),n_one=sum(s==1),fit_seconds=NA_real_,warnings='',convergence='',
  lme4_version=as.character(packageVersion('lme4')))
write.csv(result,destination,row.names=FALSE)

withCallingHandlers(tryCatch({
  if (!all(is.finite(s)) || !is.finite(sd(s)) || sd(s)==0) {
    result$status <- 'degenerate'
    warnings <- c(warnings,'Degenerate raw exponential predictor')
  } else {
    fit <- glmer(model_formula,data=d,family=binomial(link='logit'),nAGQ=1L,
      control=glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=20000),calc.derivs=TRUE))
    co <- coef(summary(fit))['similarity_raw',]
    msg <- fit@optinfo$conv$lme4$messages
    code <- fit@optinfo$conv$opt
    critical <- if(is.null(msg)) character() else msg[!grepl('boundary.*singular',msg)]
    ok <- (is.null(code)||all(code==0)) && length(critical)==0 && all(is.finite(co))
    result$status <- if(ok) 'ok' else 'nonconverged'
    result$z <- co['z value']; result$coefficient <- co['Estimate']; result$std_error <- co['Std. Error']
    result$log_likelihood <- as.numeric(logLik(fit)); result$singular <- isSingular(fit,tol=1e-4)
    result$convergence <- if(is.null(msg)) 'ok' else paste(msg,collapse=' | ')
    vc <- as.data.frame(VarCorr(fit))
    result$var_participant <- vc$vcov[vc$grp=='participant_id'][1]
    result$var_item <- vc$vcov[vc$grp=='analysis_item_id'][1]
  }
}, error=function(e) {result$status <<- 'failed'; warnings <<- c(warnings,conditionMessage(e))}),
warning=function(w) {warnings <<- c(warnings,conditionMessage(w)); invokeRestart('muffleWarning')})
result$warnings <- paste(unique(warnings),collapse=' | ')
result$fit_seconds <- proc.time()[[3]]-t0
write.csv(result,destination,row.names=FALSE)
