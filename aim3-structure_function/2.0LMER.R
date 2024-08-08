## Import Libraries

rm(list = ls())
graphics.off()
library(tidyverse)
library(plyr)
library(dplyr)
library(corrplot)
library(ggplot2)
library(cowplot)
library(readr)
library(gridExtra)
library(grid)
library(readxl)
library(here)
library(psych)
library(RColorBrewer)
library(sjPlot)
library(sjmisc)
library(sjlabelled)
library(lme4)
library(naniar)
library(reticulate)
use_python("/usr/bin/python3")
set_theme(base = theme_minimal())
# print versions pls
print(sessionInfo())


#**Data :**  cleaned data, pre processed and
#grouped into two different datasets -
#one for RSI measures and other for RSVar
#pollutants exposure measures from ABCD data.

# load cleaned and prepped  RSI data
base_dir <- "/Volumes/projects_herting/LABDOCS/Personnel"
proj_dir <- paste(base_dir,
                  "Katie/SCEHSC_Pilot/aim3",
                  sep = "/")
data_dir <- "data"
figs_dir <- "figures"
outp_dir <- "output"

df <- readRDS('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.1/01_Demographics/ABCD_5.1_demographics_full.RDS')

df <- subset(df, eventname=='baseline_year_1_arm_1')
rownames(df) <- df$src_subject_id

bmi_df <- readRDS(
  "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.0/04_Physical_Health/ABCD_5.0_Physical_Health.Rds",
  #header = TRUE,
  #row.names = 1
)

bmi_df <- subset(bmi_df, eventname=='baseline_year_1_arm_1')
rownames(bmi_df) <- bmi_df$src_subject_id

hormone_df <- readRDS(
  "/Volumes/projects_herting/LABDOCS/Personnel/Katie/kangaroo_aim1/data/SalimetricsClean.RDS",
  #header = TRUE,
  #row.names = 1
)

hormone_df <- subset(hormone_df, eventname=='baseline_year_1_arm_1')
rownames(hormone_df) <- hormone_df$src_subject_id

hormone_cols <- c(
  "filtered_dhea",
  "filtered_estradiol",
  "filtered_testosterone"
)

cov_cols <- c("demo_sex_v2_bl",
              "interview_age",
              #"ehi_y_ss_scoreb",
              "site_id_l",
              "household_income_4bins_bl",
              "race_ethnicity_c_bl",
              'rel_family_id_bl',
              "bmi_z"#,
              #"dmri_meanmotion", 
              #"rsfmri_meanmotion"
)

ppts_in_common <- union(rownames(df), rownames(hormone_df))
df1 <- cbind(df[ppts_in_common,], hormone_df[ppts_in_common,], bmi_df[ppts_in_common,])
temp_df <- df1[, c(
  hormone_cols, 
  cov_cols
)]

#complete_df <- drop_na(temp_df)

regform1 <- 'V1 ~ filtered_dhea + filtered_testosterone + (1|rel_family_id_bl:site_id_l)'
regform2 <- 'V1 ~ filtered_dhea + filtered_testosterone + filtered_estradiol + (1|rel_family_id_bl:site_id_l)'


# first we have to regress the common factors out of the hormone levels:
dhea_df <- temp_df[, c("filtered_dhea", cov_cols)]
complete_df <- drop_na(dhea_df)

dhea_model <- lm(complete_df[,c("filtered_dhea")] ~ complete_df$interview_age +
                   complete_df$race_ethnicity_c_bl +
                   complete_df$household_income_4bins_bl +
                   complete_df$bmi_z +
                   complete_df$demo_sex_v2_bl +
                   complete_df$site_id_l, na.action = na.omit) # data =)

dhea_residuals <- data.frame()
dhea_residuals <- as.data.frame(dhea_model$residuals)
rownames(dhea_residuals) <- rownames(complete_df)

t_df <- temp_df[, c("filtered_testosterone", cov_cols)]
complete_df <- drop_na(t_df)

t_model <- lm(complete_df[,c("filtered_testosterone")] ~ complete_df$interview_age +
                   complete_df$race_ethnicity_c_bl +
                   complete_df$household_income_4bins_bl +
                   complete_df$bmi_z +
                   complete_df$demo_sex_v2_bl +
                   complete_df$site_id_l, na.action = na.omit) # data =)

t_residuals <- data.frame()
t_residuals <- as.data.frame(t_model$residuals)
rownames(t_residuals) <- rownames(complete_df)

e2_df <- temp_df[, c("filtered_estradiol", cov_cols)]
complete_df <- drop_na(e2_df)

e2_model <- lm(complete_df$filtered_estradiol ~ complete_df$interview_age +
                 complete_df$race_ethnicity_c_bl +
                 complete_df$household_income_4bins_bl +
                 complete_df$bmi_z +
                 complete_df$demo_sex_v2_bl +
                 complete_df$site_id_l, na.action = na.omit) # data =)

e2_residuals <- data.frame()
e2_residuals <- as.data.frame(e2_model$residuals)
rownames(e2_residuals) <- rownames(complete_df)

# need a new df with site, family, and hormone residuals

temp1 <- union(rownames(e2_residuals), rownames(t_residuals))
all_hormone_ppts <- union(temp1, rownames(dhea_residuals))


rni_latent_x <- read.csv(
  '/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/output/rniXrsfc_lx-base.csv',
  row.names = 1
)

rni_latent_y <- read.csv(
  '/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/output/rniXrsfc_ly-base.csv',
  row.names = 1
)

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
             na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
               na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar1_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar1_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar1_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar1_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

###########################################################################
################################ RND ######################################

rnd_latent_x <- read.csv(
  '/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/output/rndXrsfc_lx-base.csv',
  row.names = 1
)

rnd_latent_y <- read.csv(
  '/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim3/output/rndXrsfc_ly-base.csv',
  row.names = 1
)

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar1_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar1_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar1_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar1_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

################################################################################
############# Latent Variable 2 ################################################
################################################################################

regform1 <- 'V2 ~ filtered_dhea + filtered_testosterone + (1|rel_family_id_bl:site_id_l)'
regform2 <- 'V2 ~ filtered_dhea + filtered_testosterone + filtered_estradiol + (1|rel_family_id_bl:site_id_l)'

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar2_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar2_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)


tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar2_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar2_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

###########################################################################
################################ RND ######################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)


tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar2_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar2_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar2_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar2_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))


################################################################################
############# Latent Variable 3 ################################################
################################################################################

regform1 <- 'V3 ~ filtered_dhea + filtered_testosterone + (1|rel_family_id_bl:site_id_l)'
regform2 <- 'V3 ~ filtered_dhea + filtered_testosterone + filtered_estradiol + (1|rel_family_id_bl:site_id_l)'

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar3_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar3_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rni_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar3_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rni_latvar3_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

###########################################################################
################################ RND ######################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_x[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)
#modelf <- lmer(sexreg,
#               na.action = na.omit, data = complete_df[f,])
#modelm <- lmer(sexreg,
#                na.action = na.omit, data = complete_df[m,])

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar3_x_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar3_x_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))

#############################################################

all_hormone_df <- cbind(
  df1[all_hormone_ppts,], 
  e2_residuals[all_hormone_ppts,],
  t_residuals[all_hormone_ppts,],
  dhea_df[all_hormone_ppts,],
  rnd_latent_y[all_hormone_ppts,]
)

model <- lmer(regform2,
               na.action = na.omit, data = all_hormone_df)
model2 <- lmer(regform1,
                na.action = na.omit, data = all_hormone_df)

tab_model(model, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar3_y_all_hormones.html',
                     sep = "/",
                     collapse = NULL))

tab_model(model2, digits = 4,show.aic = T, show.std = "std2", 
          file=paste(proj_dir,
                     outp_dir,
                     'rnd_latvar3_y_dhea_t.html.html',
                     sep = "/",
                     collapse = NULL))