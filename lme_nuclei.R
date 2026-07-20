library(dplyr)
library(lme4)
library(lmerTest)
library(emmeans)
library(writexl)

run_nuclei_lme <- function(dat, outcome, task_value, window_value, cond_levels, analysis_name = NULL, adjust_method = "bonferroni") {

  df <- dat %>%
    filter(task == task_value, window == window_value, cond %in% cond_levels) %>%
    mutate(
      task = factor(task),
      window = factor(window),
      cond = factor(cond, levels = cond_levels),
      nuclei = factor(nuclei, levels = c("Lat", "BL", "BM", "CMN")),
      sub = factor(sub),
      contact = factor(contact)
    ) %>%
    droplevels()

  contrasts(df$cond) <- contr.sum(nlevels(df$cond))
  contrasts(df$nuclei) <- contr.sum(nlevels(df$nuclei))

  model_formula <- as.formula(paste0(outcome, " ~ nuclei * cond + (1 | sub) + (1 | sub:contact)"))
  model <- lmer(model_formula, data = df, REML = FALSE)

  anova_type3 <- anova(model, type = 3)

  emm_cond_by_nuclei <- emmeans(model, ~ cond | nuclei)
  cond_effect_by_nuclei <- contrast(emm_cond_by_nuclei, method = "revpairwise", adjust = adjust_method)

  emm_nuclei_by_cond <- emmeans(model, ~ nuclei | cond)
  nuclei_effect_by_cond <- pairs(emm_nuclei_by_cond, adjust = adjust_method)

  emm_full <- emmeans(model, ~ cond * nuclei)
  interaction_contrast <- contrast(
    emm_full,
    interaction = c("revpairwise", "pairwise"),
    adjust = adjust_method
  )

  print(summary(model))
  print(anova_type3)

  cat("\nSingular fit:\n")
  print(isSingular(model, tol = 1e-4))

  cat("\nCondition effect within each nucleus:\n")
  print(cond_effect_by_nuclei)

  cat("\nNuclei comparisons within each condition:\n")
  print(nuclei_effect_by_cond)

  cat("\nCondition-effect differences between nuclei:\n")
  print(interaction_contrast)

  list(
    data = df,
    formula = model_formula,
    model = model,
    model_summary = summary(model),
    anova = as.data.frame(anova_type3),
    emm_cond_by_nuclei = as.data.frame(emm_cond_by_nuclei),
    cond_effect_by_nuclei = as.data.frame(cond_effect_by_nuclei),
    emm_nuclei_by_cond = as.data.frame(emm_nuclei_by_cond),
    nuclei_effect_by_cond = as.data.frame(nuclei_effect_by_cond),
    interaction_contrast = as.data.frame(interaction_contrast),
    singular = isSingular(model, tol = 1e-4)
  )
}
