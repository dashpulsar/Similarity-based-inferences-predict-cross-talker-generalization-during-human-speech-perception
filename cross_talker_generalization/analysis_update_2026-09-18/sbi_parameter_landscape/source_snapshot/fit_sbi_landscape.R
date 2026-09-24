args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L)
suppressPackageStartupMessages(library(lme4))
d <- read.csv(args[1],stringsAsFactors=FALSE)
grid <- read.csv(args[2])
stopifnot(all(d$fold!=0),all(is.finite(d$raw_distance)))
for (v in c('participant_id','analysis_item_id','test_talker_id')) d[[v]]<-factor(d[[v]])
for (i in seq_len(nrow(grid))) {
    k <- grid$k[i]
    # expm1 is an exact affine shift of exp; standardization removes the shift.
    # k=0 denotes a separately labeled small-k standardized limit, not constant similarity.
    s <- if(k==0) -d$raw_distance else expm1(-k*d$raw_distance)
    t0 <- proc.time()[[3]]
    warnings <- character()
    result <- data.frame(k=k,kind=if(k==0) 'linear_limit' else 'grid',status='failed',
        z=NA_real_,log_likelihood=NA_real_,coefficient=NA_real_,std_error=NA_real_,singular=NA,
        var_participant=NA_real_,var_item=NA_real_,var_talker=NA_real_,
        n_rows=nrow(d),participants=nlevels(d$participant_id),n_word_responses=sum(d$response_correct+d$response_incorrect),
        predictor_shifted_mean=mean(s),predictor_sd=sd(s),fit_seconds=NA_real_,warnings='',convergence='',
        lme4_version=as.character(packageVersion('lme4')))
    fit <- NULL
    if (is.finite(sd(s)) && sd(s)>0) {
        d$similarity_z <- (s-mean(s))/sd(s)
        fit <- withCallingHandlers(tryCatch(glmer(
            cbind(response_correct,response_incorrect)~similarity_z+
            (1|participant_id)+(1|analysis_item_id)+(1|test_talker_id),
            data=d,family=binomial(link='logit'),nAGQ=1L,
            control=glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=20000),calc.derivs=TRUE)),
            error=function(e){warnings<<-c(warnings,conditionMessage(e));NULL}),
            warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
    } else { warnings<-c(warnings,'Degenerate predictor') }
    if(!is.null(fit)) {
        co <- coef(summary(fit))['similarity_z',]
        msg <- fit@optinfo$conv$lme4$messages
        code <- fit@optinfo$conv$opt
        critical <- if(is.null(msg)) character() else msg[!grepl('boundary.*singular',msg)]
        ok <- (is.null(code)||all(code==0)) && length(critical)==0 && all(is.finite(co))
        result$status<-if(ok) 'ok' else 'nonconverged'
        result$z<-co['z value'];result$coefficient<-co['Estimate'];result$std_error<-co['Std. Error']
        result$log_likelihood<-as.numeric(logLik(fit));result$singular<-isSingular(fit,tol=1e-4)
        result$convergence<-if(is.null(msg)) 'ok' else paste(msg,collapse=' | ')
        vc<-as.data.frame(VarCorr(fit))
        result$var_participant<-vc$vcov[vc$grp=='participant_id'][1]
        result$var_item<-vc$vcov[vc$grp=='analysis_item_id'][1]
        result$var_talker<-vc$vcov[vc$grp=='test_talker_id'][1]
    }
    result$warnings<-paste(unique(warnings),collapse=' | ')
    result$fit_seconds<-proc.time()[[3]]-t0
    write.table(result,args[3],sep=',',row.names=FALSE,col.names=!file.exists(args[3]),append=file.exists(args[3]),quote=TRUE)
}
