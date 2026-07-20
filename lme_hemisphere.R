library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(broom.mixed)

run_hemisphere_lme <- function(dat, outcome, random_effects = "(1 | sub)", adjust_method = "bonferroni") {
  
  # Convert grouping and fixed-effect variables to factors
  dat <- dat %>%
    mutate(
      across(any_of(c("sub", "channel_idx", "pair_idx")), factor),
      hemisphere = factor(hemisphere, levels = c("L", "R")),
      region = factor(region, levels = c("l", "m")),
      time = factor(time, levels = c("early", "late"))
    ) %>%
    droplevels()
  
  # Sum-to-zero contrasts for Type III tests
  contrasts(dat$hemisphere) <- contr.sum(nlevels(dat$hemisphere))
  contrasts(dat$region) <- contr.sum(nlevels(dat$region))
  contrasts(dat$time) <- contr.sum(nlevels(dat$time))
  
  # Construct and fit the linear mixed-effects model
  fixed_formula <- paste0(outcome, " ~ region * time * hemisphere")
  model_formula <- as.formula(paste(fixed_formula, random_effects, sep = " + "))
  model <- lmer(model_formula, data = dat, REML = FALSE)
  
  # Omnibus Type III tests
  anova_type3 <- anova(model, type = 3)
  
  # Estimated marginal means for two-way interaction contrasts
  emm_region_hemi <- emmeans(model, ~ region * hemisphere)
  emm_time_hemi <- emmeans(model, ~ time * hemisphere)
  emm_time_region <- emmeans(model, ~ time * region)
  
  interaction_region_hemi <- contrast(
    emm_region_hemi,
    interaction = c("pairwise", "pairwise"),
    adjust = adjust_method
  )
  
  interaction_time_hemi <- contrast(
    emm_time_hemi,
    interaction = c("pairwise", "pairwise"),
    adjust = adjust_method
  )
  
  interaction_time_region <- contrast(
    emm_time_region,
    interaction = c("pairwise", "pairwise"),
    adjust = adjust_method
  )
  
  # Simple-effect comparisons
  simple_region <- pairs(
    emmeans(model, ~ region | time * hemisphere),
    adjust = adjust_method
  )
  
  simple_time <- pairs(
    emmeans(model, ~ time | region * hemisphere),
    adjust = adjust_method
  )
  
  simple_hemisphere <- pairs(
    emmeans(model, ~ hemisphere | region * time),
    adjust = adjust_method
  )
  
  # All region × time comparisons within each hemisphere
  region_time_by_hemisphere <- pairs(
    emmeans(model, ~ region * time | hemisphere),
    adjust = adjust_method
  )
  
  # Return all results
  list(
    data = dat,
    formula = model_formula,
    model = model,
    model_summary = summary(model),
    anova_type3 = anova_type3,
    fixed_effects = broom.mixed::tidy(model, effects = "fixed", conf.int = TRUE),
    interaction_region_hemisphere = summary(interaction_region_hemi),
    interaction_time_hemisphere = summary(interaction_time_hemi),
    interaction_time_region = summary(interaction_time_region),
    simple_region = summary(simple_region),
    simple_time = summary(simple_time),
    simple_hemisphere = summary(simple_hemisphere),
    region_time_by_hemisphere = summary(region_time_by_hemisphere)
  )
}
