# Exercise the actual two scaling expressions in the landscape fitter.
source_lines <- readLines('cross_talker_generalization/R/fit_sbi_landscape.R')
expressions <- parse(text=source_lines[grepl('^    (a|s) <-',source_lines)])
stopifnot(length(expressions)==2)
standardize <- function(x) (x-mean(x))/sd(x)
count <- 0L
for (distances in list(c(0,1,5,20,45),c(1000,1001,1005,1020,1045))) {
    for (k in c(0,1e-12,1e-6,.001,.05,2)) {
        d <- data.frame(raw_distance=distances)
        eval(expressions)
        stopifnot(all(is.finite(s)),sd(s)>0)
        z <- standardize(s)
        stopifnot(abs(mean(z))<1e-10,abs(sd(z)-1)<1e-10)
        if(k==0) stopifnot(max(abs(z-standardize(-distances)))<1e-12)
        # Direct exponential is a suitable reference only where representable.
        naive <- exp(-k*distances)
        if(k>=1e-6 && is.finite(sd(naive)) && sd(naive)>1e-100) {
            stopifnot(max(abs(z-standardize(naive)))<1e-7)
        }
        if(k>0 && k<1e-10) stopifnot(max(abs(z-standardize(-distances)))<1e-8)
        count<-count+1L
    }
}
cat(count,'scaling cases passed, including cancellation and underflow cases.\n')
