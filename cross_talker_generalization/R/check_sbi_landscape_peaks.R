# Numerical cross-check of two preidentified candidate peaks using another inner solver.
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1)
suppressPackageStartupMessages(library(lme4))
out<-args[1]
timing<-read.csv(file.path(out,'preparation','dtw_timings.csv'))
rows<-list()
for(tau in c(.5,1)) {
    idx<-which(timing$tau==tau)
    d<-read.csv(file.path(out,'preparation',basename(timing$training_file[idx])))
    candidates<-read.csv(file.path(out,sprintf('tau_%03d_fits.csv',idx-1)))
    candidates<-candidates[candidates$kind=='grid' & candidates$status=='ok' & !candidates$singular,]
    best<-candidates[which.max(candidates$log_likelihood),]
    for(v in c('participant_id','analysis_item_id','test_talker_id'))d[[v]]<-factor(d[[v]])
    stopifnot(all(d$fold!=0))
    a <- -best$k*(d$raw_distance-min(d$raw_distance))
    s <- if(max(abs(a))<.1)expm1(a) else exp(a)
    d$similarity_z<-(s-mean(s))/sd(s)
    f<-glmer(cbind(response_correct,response_incorrect)~similarity_z+test_talker_id+
        (1|participant_id)+(1|analysis_item_id),data=d,family=binomial(),nAGQ=1,
        control=glmerControl(optimizer='Nelder_Mead',optCtrl=list(maxfun=100000)))
    c<-coef(summary(f))['similarity_z',]
    messages<-f@optinfo$conv$lme4$messages
    rows[[length(rows)+1]]<-data.frame(tau=tau,k=best$k,bobyqa_z=best$z,nelder_mead_z=c['z value'],
        bobyqa_log_likelihood=best$log_likelihood,nelder_mead_log_likelihood=as.numeric(logLik(f)),
        convergence=if(is.null(messages))'ok' else paste(messages,collapse=' | '),
        singular=isSingular(f,tol=1e-4))
}
write.csv(do.call(rbind,rows),file.path(out,'peak_solver_check.csv'),row.names=FALSE)
