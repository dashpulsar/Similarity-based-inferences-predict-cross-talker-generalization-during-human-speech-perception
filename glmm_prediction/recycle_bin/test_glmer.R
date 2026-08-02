library(lme4)
df <- read.csv("debug_train_agg.csv")
df$Keyword <- factor(df$Keyword)
df$TestTalker <- factor(df$TestTalker)
df$SubjectID <- factor(df$SubjectID)

print(head(df))
print(str(df))

tryCatch({
    model <- glmer(cbind(numCorrect, numIncorrect) ~ 1 + similarity_scaled + (1|TestTalker) + (1|Keyword), 
          data=df, family=binomial(link="logit"), 
          control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=1e5)))
    print(summary(model))
}, error=function(e){ print("GLMER ERROR:"); print(e) })
