## Import Libraries
rm(list = ls())
graphics.off()
library(tidyverse)
library(ExPosition)
library(TExPosition)
library(PTCA4CATA)
library(data4PCCAR)
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
#df <- read.csv(paste(proj_dir,
#                     data_dir,
#                     "data_qcd.csv",
#                    sep = "/",
#                     collapse = NULL),
#               header = TRUE,
#               row.names = 1)

df <- readRDS('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.1/01_Demographics/ABCD_5.1_demographics_full.RDS')

df <- subset(df, eventname=='baseline_year_1_arm_1')
rownames(df) <- df$src_subject_id

img_cov <- readRDS('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.1/03_Imaging/ABCD_5.1_Covariates_Imaging.RDS')
img_cov <- subset(img_cov, eventname=='baseline_year_1_arm_1')
rownames(img_cov) <- img_cov$src_subject_id
brain <- read.csv(paste(proj_dir,
                        data_dir,
                        "rsi_and_rsfc_data_base-qcd.csv",
                        sep = "/",
                        collapse = NULL),
                  header = TRUE,
                  row.names = 1)
bmi_df <- readRDS(
  "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.0/04_Physical_Health/ABCD_5.0_Physical_Health.Rds",
  #header = TRUE,
  #row.names = 1
)

bmi_df <- subset(bmi_df, eventname=='baseline_year_1_arm_1')
rownames(bmi_df) <- bmi_df$src_subject_id

rnd_cols <- colnames(brain[, grep(pattern = "dmri_rsirnd.*",
                                  colnames(brain))])
rsfmri_cols <- colnames(brain[, grep(pattern = "rsfmri_c_.*",
                                     colnames(brain))])
cov_cols <- c("demo_sex_v2_bl",
              "interview_age",
              "handedness_baseline",
              "site_id_l",
              "mri_info_manufacturer",
              "household_income_4bins_bl",
              "race_ethnicity_c_bl",
              "bmi_z",
              "dmri_meanmotion", 
              "rsfmri_meanmotion"
)

ppts_in_common <- union(rownames(df), rownames(brain))
ppts_in_common <- union(ppts_in_common, rownames(bmi_df))
df1 <- cbind(df[ppts_in_common,], brain[ppts_in_common,], bmi_df[ppts_in_common,], img_cov[ppts_in_common,])
temp_df <- df1[, c(
  rnd_cols, 
  rsfmri_cols, 
  cov_cols
)]

complete_df <- drop_na(temp_df)



#load cleaned and prepped data
df_rsfc <- complete_df[, rsfmri_cols]
df_rsi <- complete_df[, rnd_cols]
df_covariates <- complete_df[, cov_cols]


## Performing residualization
###  MLRM for RSI data

# dependent variables
Group1.Y <- as.matrix(df_rsi)
dim(Group1.Y)

# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1 <- lm(Group1.Y ~ Group1.X$interview_age +
                  Group1.X$race_ethnicity_c_bl +
                  Group1.X$household_income_4bins_bl +
                  Group1.X$bmi_z +
                  Group1.X$demo_sex_v2_bl +
                  Group1.X$handedness_baseline +
                  Group1.X$mri_info_manufacturer +
                  Group1.X$site_id_l +
                  Group1.X$dmri_meanmotion, na.action = na.omit) # data =)



# R.square -> how well the model explains the variation
# in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals <- as.data.frame(lm.Group1$residuals)

png(paste(proj_dir, figs_dir, "rnd-brain_residuals-base.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group1), residuals(lm.Group1))
dev.off()

#### Measure goodness of fit

#{r echo=TRUE, fig.height=10, fig.width=12}
# for one outcome variable

# for matrix of outcome variables
# loop the variables to obtain qq plot


#hist(as.data.frame(Group1_residuals))
# evaluation by predicton

f1 <- fitted(lm.Group1)
r1 <- residuals(lm.Group1)


for (i in seq_len(ncol(f1))) {
  plot(f1[, i], r1[, i],
       main = paste0("fitted VS residual:   ", colnames(f1)[i], sep = ""))
}





### MLRM for rsFC data

#{r echo=TRUE}

Group2.Y <- as.matrix(df_rsfc)


lm.Group2 <- lm(Group2.Y ~ Group1.X$interview_age +
                  Group1.X$race_ethnicity_c_bl +
                  Group1.X$household_income_4bins_bl +
                  Group1.X$bmi_z +
                  Group1.X$demo_sex_v2_bl +
                  Group1.X$handedness_baseline +
                  Group1.X$mri_info_manufacturer +
                  Group1.X$site_id_l +
                  Group1.X$rsfmri_meanmotion, na.action = na.omit)

Group2_residuals <- data.frame()
Group2_residuals <- as.data.frame(lm.Group2$residuals)

#### Measure goodness of fit

#{r echo=TRUE}



#hist(as.data.frame(Group2_residuals))

png(paste(proj_dir, figs_dir, "rnd-rsvar_pollution_residuals-base.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
plot(fitted(lm.Group2), residuals(lm.Group2))
dev.off()

f <- fitted(lm.Group2)
r <- residuals(lm.Group2)




# colors set by site
color_site <- brewer.pal(4, "PuOr")
color_site <- colorRampPalette(color_site)(22)
names(color_site) <- levels(unique(df_covariates$site))
colScale <- scale_colour_manual(name = "site", values = color_site)




# Color for rows
col4row  <- matrix(nrow = nrow(df_covariates), ncol = 1)

for (i in seq_along(unique(df_covariates$site))) {
  col4row[which(df_covariates$site == unique(df_covariates$site)[i])] <- color_site[i]
}


# Colors set by area
color_area <- brewer.pal(4, "PuOr")
color_area <- colorRampPalette(color_area)(3)
names(color_site) <- levels(unique(df_covariates$reshist_addr1_urban_area))
colScale <- scale_colour_manual(name = "area", values = color_site)

col4row_area  <- matrix(nrow = nrow(df_covariates), ncol = 1)

for (i in seq_along(unique(df_covariates$reshist_addr1_urban_area))) {
  col4row_area[which(df_covariates$reshist_addr1_urban_area == unique(df_covariates$reshist_addr1_urban_area)[i])] <- color_area[i]
}


# colors set by age
df_covariates$age <- cut(df_covariates$interview_age,
                         breaks = c(0, 114, 120, 126, 133),
                         labels = c("9-9.5 yrs",
                                    "9.5-10 yrs",
                                    "10-10.5 yrs",
                                    "10.5-11 yrs"))






## PLSC Analysis

### 1. Correlation plots

#{r fig.height= 5, fig.width=20, fig.align="center"}

#Input data for PLSC
data1 <- as.data.frame(Group1_residuals)
data2 <- as.data.frame(Group2_residuals)


# Compute the covariance matrix
print("correlations time!")
XY.cor.pearson <- cor(data2, data1)

png(paste(proj_dir, figs_dir, "rndXrsvar_corr-base.png", sep = "/"),
    width = 10, height = 5, res = 600, units = "in")
corrplot(XY.cor.pearson, method = "color", tl.cex = 0.7, 
         cl.pos = "r", cl.cex = .7, mar = c(0, 0, 1, 0),
         title = "Pearson Correlation", addCoefasPercent = FALSE,
         col.lim = c(min(XY.cor.pearson), max(XY.cor.pearson)),
         col = colorRampPalette(c("darkred",
                                  "white",
                                  "midnightblue"))(100))
dev.off()

#XY.cor.kendall <- cor(data1, data2, method = "kendall")
#png(paste(proj_dir, figs_dir, "rndXrsvar_ckorr-base.png", sep = "/"),
#    width = 10, height = 5, res = 600, units = "in")
#corrplot(t(XY.cor.kendall), method = "color", tl.cex = 0.7, 
#          cl.pos = "r", cl.cex = .7, title = "Kendall Correlation",
#          addCoefasPercent = FALSE, mar = c(0, 0, 1, 0), 
#          col.lim = c(min(XY.cor.kendall), max(XY.cor.kendall)),
#          col = colorRampPalette(c("darkred", "white", "midnightblue"))(100))
#dev.off()



### 2. Package details


tepPLS(data1, data2,
       center1 = TRUE,
       scale1 = "SS1",
       center2 = TRUE,
       scale2 = "SS1",
       DESIGN = NULL,
       make_design_nominal = TRUE,
       graphs = TRUE,
       k = 0)

# DATA1 : Data matrix 1 (X)
# DATA2 : Data matrix 2 (Y)
# center1 : a boolean, vector, or string to center DATA1. See expo.scale for details. # nolint
# scale1 : a boolean, vector, or string to scale DATA1. See expo.scale for details. # nolint
# center2 : a boolean, vector, or string to center DATA2. See expo.scale for details. # nolint
# scale2 : a boolean, vector, or string to scale DATA2. See expo.scale for details. # nolint
# DESIGN : a design matrix to indicate if rows belong to groups. # nolint
# make_design_nominal	: a boolean. If TRUE (default), DESIGN is a vector that
# indicates groups (and will be dummy-coded). If FALSE, DESIGN is a dummy-coded matrix. # nolint
# graphs : a boolean. If TRUE (default), graphs and plots are provided (via tepGraphs) # nolint
# k	: number of components to return.



# {r echo=TRUE}

data.design <- df_covariates$site_id_l


# make the design into a vector
data.design.vec <- as.matrix(data.design)

rownames(data.design.vec) <- df_covariates$subjectkey

print("pls let's goooooo!")
# Applying PLSC function to the prepped data
pls.res <- tepPLS(data1, data2,
                  DESIGN = data.design,
                  make_design_nominal = TRUE,
                  graphs = FALSE)



### 3. Scree Plot

# The scree plot shows a weird pattern because the null
# hypothesis is that there is no correlation between the tables (Null = 0)
# Hence eigenvalues greater than zero become significant.

# The results of the permutation test gives us the eigenvalues.

# {r echo=TRUE}
# no.of eigenvalues
nL <- min(ncol(data1), ncol(data2))


# Applying permutation test to the input data for PLSC
resPerm4PLSC <- perm4PLSC(data1, # First Data matrix
                          data2, # Second Data matrix
                          permType = "byColumns",
                          nIter = 10000 # How many iterations
)
print(resPerm4PLSC)

scree_df <- data.frame(row.names = 1:68)
scree_df$eigens <- pls.res$TExPosition.Data$eigs
scree_df$pEigens <- resPerm4PLSC$pEigenvalues


write.csv(scree_df,
          paste(proj_dir, outp_dir,
                paste("rndXrsfc-scree_values_",
                      resPerm4PLSC$pOmnibus,
                      "-base.csv",
                      sep = ""),
                sep = "/"))

print("screeeeeeeeee")
# obtaining the scree plot
png(paste(proj_dir, figs_dir, "rndXrsfc-PLSC_scree-base.png", sep = "/"),
    width = 5, height = 5, res = 600, units = "in")
screeeeee <- PlotScree(ev = pls.res$TExPosition.Data$eigs,
                       title = "PLSC- Scree Plot",
                       p.ev = resPerm4PLSC$pEigenvalues,
                       plotKaiser = TRUE,
                       color4Kaiser = ggplot2::alpha("darkorchid4", .5),
)
dev.off()
screeeeee



### 4. Looking at the First Pair of Latent variables by site



# ploting the first latent variable of data1(X) and first
#latent variables of data2(Y). We are tryingto see if these
#two latent variables are similar or not.

# first pair of latent variables:

latvar.1 <- cbind(pls.res$TExPosition.Data$lx[, 1],
                  pls.res$TExPosition.Data$ly[, 1])
colnames(latvar.1) <- c("Lx 1", "Ly 1")

latvar.2 <- cbind(pls.res$TExPosition.Data$lx[, 2],
                  pls.res$TExPosition.Data$ly[, 2])
colnames(latvar.2) <- c("Lx 2", "Ly 2")

# compute means
lv.1.group <- getMeans(latvar.1, data.design.vec)
lv.2.group <- getMeans(latvar.2, data.design.vec)

col4Means <- as.matrix(color_site)
rownames(col4Means) <- rownames(lv.1.group)

print("bootstrapping...")
# compute bootstrap - for confidence intervals
lv.1.group.boot <- Boot4Mean(latvar.1, data.design.vec)
colnames(lv.1.group.boot$BootCube) <- c("Lx 1", "Ly 1")

rownames(latvar.1) <- df_covariates$subjectkey

# plotiing the factor Maps
png(paste(proj_dir, figs_dir, "rndXrsfc-LD1_factor_map-base.png", sep = "/"),
    width = 10, height = 10, res = 600, units = "in")
plot.lv1 <- createFactorMap(latvar.1,
                            col.points = col4row,
                            col.labels = col4row,
                            alpha.points = 0.7,
                            force = 0.01
                            
)
plot1.mean <- createFactorMap(lv.1.group,
                              col.points = col4Means,
                              col.labels = col4Means,
                              cex = 4,
                              pch = 17,
                              force = 0.1,
                              alpha.points = 0.8)

plot1.meanCI <- MakeCIEllipses(lv.1.group.boot$BootCube[, c(1:2), ], # get the first two components
                               col = col4Means[rownames(lv.1.group.boot$BootCube)],
                               names.of.factors = c("Lx 1", "Ly 1")
)
plot1 <- plot.lv1$zeMap_background + plot.lv1$zeMap_dots+ plot.lv1$zeMap_text + plot1.mean$zeMap_dots + plot1.mean$zeMap_text + plot1.meanCI 

ggsave(plot1, file=paste(proj_dir, figs_dir, "rndXrsfc-LD1_factor_map-base.png", 
                         sep="/"),
       width = 10, height = 10, units = "in", dpi=600)


# check for the outlier
outlier <- subset(latvar.1, rowSums(latvar.1 > 0.3) > 0)

indx <- which(rownames(latvar.1) == rownames(outlier))
print(rownames(outlier))

# latent dimension # 2
rownames(col4Means) <- rownames(lv.2.group)

# compute bootstrap - for confidence intervals
lv.2.group.boot <- Boot4Mean(latvar.2, data.design.vec)
colnames(lv.2.group.boot$BootCube) <- c("Lx 2", "Ly 2")

rownames(latvar.2) <- df_covariates$subjectkey

# plotiing the factor Maps

plot.lv2 <- createFactorMap(latvar.2,
                            col.points = col4row,
                            col.labels = col4row,
                            alpha.points = 0.7,
                            force = 0.01
                            
)

plot2.mean <- createFactorMap(lv.2.group,
                              col.points = col4Means,
                              col.labels = col4Means,
                              cex = 4,
                              pch = 17,
                              force = 0.1,
                              alpha.points = 0.8)

plot2.meanCI <- MakeCIEllipses(lv.2.group.boot$BootCube[,c(1:2),], # get the first two components
                               col = col4Means[rownames(lv.2.group.boot$BootCube)],
                               names.of.factors = c("Lx 2", "Ly 2")
)

plot2 <- plot.lv2$zeMap_background + plot.lv2$zeMap_dots+ plot.lv2$zeMap_text + plot2.mean$zeMap_dots + plot2.mean$zeMap_text + plot2.meanCI 

ggsave(plot2, file=paste(proj_dir, figs_dir, "rndXrsfc-LD2_factor_map-base.png", 
                         sep="/"),
       width = 10, height = 10, units = "in", dpi=600)

# check for the outlier
outlier <- subset(latvar.2, rowSums(latvar.2 > 0.3) > 0)

indx <- which(rownames(latvar.2)== rownames(outlier))
print(rownames(outlier))


### 5. Obtaining Column factor scores

# {r echo=TRUE}
### Column Factor scores of the 1st component of data1 representing RSI measures and data2 representing rsvar pollutants

#Fi:column factor scores for data1(RSI measures) or
#Loadings of data1(RSI measures)
Fi <- pls.res$TExPosition.Data$fi

#Fi:column factor scores for data2(rsvar pollutants exposure)
#or Loadings of data2(rsvar pollutants exposure)
Fj <- pls.res$TExPosition.Data$fj

# Generating loadings map of Fi
p.loadings <- createFactorMap(Fi,
                              axis1 = 1,
                              axis2 = 2,
                              display.points = TRUE,
                              display.labels = TRUE,
                              #col.points = col4column_rsi,
                              #col.labels = col4column_rsi,
                              title = "Loadings of Columns RSI",
                              pch = 20,
                              cex = 3,
                              text.cex = 3,
)

label4map <- createxyLabels.gen(x_axis = 1, y_axis = 2,
                                lambda = pls.res$TExPosition.Data$eigs,
                                tau = pls.res$TExPosition.Data$t
)

p.plot <- p.loadings$zeMap + label4map
ggsave(p.plot, file=paste(proj_dir, figs_dir, "rndXrsfc-brain_factors-base.png", 
                          sep="/"), 
       width = 10, height = 10, units = "in", dpi=600)

# Generating loadings map of Fj

q.loadings <- createFactorMap(Fj,
                              axis1 = 1,
                              axis2 = 2,
                              display.points = TRUE,
                              display.labels = TRUE,
                              #col.points = col4air,
                              #col.labels = col4air,
                              title = "Loadings of Functional Connectivity",
                              pch = 20,
                              cex = 3,
                              text.cex = 4,
)

label4map <- createxyLabels.gen(x_axis = 1, y_axis = 2,
                                lambda = pls.res$TExPosition.Data$eigs,
                                tau = pls.res$TExPosition.Data$t
)

q.plot <- q.loadings$zeMap + label4map 
ggsave(q.plot, file=paste(proj_dir, figs_dir, "rndXrsfc_factors-base.png", 
                          sep="/"),
       width = 10, height = 10, units = "in", dpi=600)

### 5. Column Loadings

# These are the loadings obtained after performing SVD(R) on correlation matrix.


# {r echo=TRUE}

#generating bar plots for loadings of RSI data table
P.data1 <- pls.res$TExPosition.Data$pdq$p


plot_P.data1 <- PrettyBarPlot2(
  bootratio = P.data1[, 1],
  threshold = 0,
  ylim = NULL,
  #color4bar = col4rsi,
  color4ns = "gray75",
  plotnames = TRUE,
  main = "Loadings of RSI variables",
  ylab = " P Loadings ",
  horizontal = TRUE)

plot_P.data1
ggsave(plot_P.data1, file=paste(proj_dir, figs_dir, "rnd-brain_P-base.png", 
                                sep = "/"), 
       width = 10, height = 10, units = "in", dpi=600)

#generating bar plots for loadings of rsvar  data table
Q.data2 <- pls.res$TExPosition.Data$pdq$q


plot_Q.data2 <- PrettyBarPlot2(
  bootratio = Q.data2[,1],
  threshold = 0,
  ylim = NULL,
  #color4bar = col4air,
  color4ns = "white",
  plotnames = TRUE,
  main = "Loadings of Functional Fluctuations",
  ylab = " Q Loadings ")

plot_Q.data2
ggsave(plot_Q.data2, file=paste(proj_dir, figs_dir, "rnd-rsfc_Q-base.png", 
                                sep = "/"), 
       width = 10, height = 10, units = "in", dpi=600)

# saving the loading table into excel file.
rsvar_loadings <- as.data.frame(Q.data2)
rnd_loadings <- as.data.frame(P.data1)


### 6. Inference Bootstrap


# The Bootstrap ratio barplot show that the
#contributions are significantly stable.
# {r echo=TRUE}
#Looking into what the resBootPLSC is giving us
print("more boootstrapping...")
resBoot4PLSC <- Boot4PLSC(data1, # First Data matrix
                          data2, # Second Data matrix
                          nIter = 10000, # How many iterations
                          Fi = pls.res$TExPosition.Data$fi,
                          Fj = pls.res$TExPosition.Data$fj,
                          nf2keep = sum(resPerm4PLSC$pEigenvalues < 0.01),
                          critical.value = 2.5,
                          # To be implemented later
                          # has no effect currently
                          alphaLevel = 1)

resBoot4PLSC

# {r echo=TRUE}
BR.I <- resBoot4PLSC$bootRatios.i
BR.J <- resBoot4PLSC$bootRatios.j

# saving the bootstrap rations into excel

rsvar_bootstrap <-  as.data.frame(BR.J)
rnd_bootstrap <-  as.data.frame(BR.I)

rsvar_res <- cbind.data.frame(rsvar_loadings, rsvar_bootstrap)
rnd_res <- cbind.data.frame(rnd_loadings, rnd_bootstrap)


write.csv(rsvar_res, paste(proj_dir,
                           outp_dir,
                           "rnd-rsfc_Q-base.csv",
                           sep = "/"), row.names = TRUE)

write.csv(rnd_res, paste(proj_dir,
                         outp_dir,
                         "rnd-brain_P-base.csv", 
                         sep = "/"), row.names = TRUE)

if (resPerm4PLSC$pOmnibus < 0.01) {
  for (i in seq_along(resPerm4PLSC$pEigenvalues)) {
    if (resPerm4PLSC$pEigenvalues[i] < 0.01) {
      laDim <- i
      print(laDim)
      png(paste(proj_dir, figs_dir, paste("rndXrsfc-LD", 
                                          i, 
                                          "_brain_saliences-base.png", 
                                          sep=""), sep = "/"),
          width = 10, height = 10, res = 600, units = "in")
      ba001.BR1.I <- PrettyBarPlot2(BR.I[,laDim],
                                    threshold = 2.5,
                                    font.size = 4,
                                    color4ns = "gray85",
                                    #color4bar = col4rsi, # we need hex code
                                    ylab = "Bootstrap ratios"
                                    #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
      ) + ggtitle(paste0("Latent dimension ",
                         laDim),
                  subtitle = "Cortical Anisotropic Intracellular Diffusion Saliences")
      ba001.BR1.I
      ggsave(ba001.BR1.I, file=paste(proj_dir, figs_dir, paste("rndXrsfc-LD", 
                                                               i, 
                                                               "_brain_saliences-base.png", 
                                                               sep=""), sep = "/"), 
             width = 10, height = 10, units = "in", dpi=600)
      
      
      png(paste(proj_dir, figs_dir, paste("rndXrsfc-LD", 
                                          i, 
                                          "_rsfc_saliences-base.png", 
                                          sep=""), sep = "/"),
          width = 10, height = 10, res = 600, units = "in")
      ba001.BR1.J <- PrettyBarPlot2(BR.J[,laDim],
                                    threshold = 2.5,
                                    font.size = 4,
                                    color4ns = "gray85", 
                                    #color4bar = gplots::col2hex(col4air),
                                    ylab = "Bootstrap ratios"
                                    #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
      ) + ggtitle(paste0("Latent dimension ", laDim), subtitle = "Functional Fluctuations Saliences (loadings)")
      ba001.BR1.J
      ggsave(ba001.BR1.J, file=paste(proj_dir, figs_dir, paste("rndXrsfc-LD", 
                                                               i, 
                                                               "_rsfc_saliences-base.png", 
                                                               sep=""), sep = "/"), 
             width = 10, height = 10, units = "in", dpi=600)
      
    }
  }
}

write.csv(pls.res$TExPosition.Data$lx, paste(proj_dir,
                                             outp_dir,
                                             "rndXrsfc_lx-base.csv",
                                             sep = "/"), row.names = TRUE)

write.csv(pls.res$TExPosition.Data$ly, paste(proj_dir,
                                             outp_dir,
                                             "rndXrsfc_ly-base.csv",
                                             sep = "/"), row.names = TRUE)