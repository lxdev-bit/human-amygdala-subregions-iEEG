# All linear mixed-effects model analyses were performed using R scripts. 
# The analytical procedures were identical across analyses, except for the fixed-effect variables of interest. 
# The specific model used for each index is described in the Methods section. 
# The R code for the analysis of z-scored power is provided below as an example:

```r
dat <- read_excel("condxpower.xlsx")

dat <- dat %>%
  mutate(
    sub = factor(sub),
    channel_idx = factor(channel_idx),
    region = factor(region, levels = c("l", "m")),
    cond = factor(cond, levels = c("neutral", "fearful", "shape")),
    timewin = factor(timewin, levels = c("early", "late")),
    hemisphere = factor(hemisphere, levels = c("L", "R"))
  )

m <- lmer(
  zpower ~ region * cond + (1 | sub) + (1 | sub:channel),
  data = dat,
  REML = FALSE
)

summary(m)
anova(m)


dat_fn <- dat %>%
  filter(cond %in% c("neutral", "fearful")) %>%
  droplevels()

contrasts(dat_fn$region)  <- contr.sum(2)
contrasts(dat_fn$timewin) <- contr.sum(2)
contrasts(dat_fn$cond)    <- contr.sum(2)

m_fn <- lmer(
  zpower ~ region * timewin * cond + (1 | sub) + (1 | sub:channel),
  data = dat_fn,
  REML = FALSE
)

anova(m_fn, type = 3)
summary(m_fn)

emm_cond <- emmeans(m_fn, ~ cond | region * timewin)
pairs(emm_cond, adjust = "bonferroni")

emm_diff <- emmeans(m_fn, ~ cond * region | timewin)

contrast(
  emm_diff,
  interaction = c("revpairwise", "revpairwise"),
  by = "timewin",
  adjust = "bonferroni"
)


dat_early_fear <- dat %>%
  filter(
    timewin == "early",
    cond %in% c("fearful", "neutral")
  )

m_early <- lmer(
  zpower ~ cond * region + (1 | sub) + (1 | sub:channel),
  data = dat_early_fear,
  REML = FALSE
)

summary(m_early)
anova(m_early)

emm_early <- emmeans(m_early, ~ cond * region)
pairs(emm_early, adjust = "bonferroni")


dat_late_fear <- dat %>%
  filter(
    timewin == "late",
    cond %in% c("fearful", "neutral")
  )

m_late <- lmer(
  zpower ~ cond * region + (1 | sub) + (1 | sub:channel),
  data = dat_late_fear,
  REML = FALSE
)

summary(m_late)
anova(m_late)

emm_late <- emmeans(m_late, ~ cond * region)
pairs(emm_late, adjust = "bonferroni")
```
