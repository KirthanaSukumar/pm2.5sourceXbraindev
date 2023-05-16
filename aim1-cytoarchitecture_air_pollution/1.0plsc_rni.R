## Import Libraries

rm(list = ls())
graphics.off()
library(tidyverse)
library(ExPosition)
install.packages("TInPosition") # if needed
library(TExPosition)
library(TInPosition)
library(PTCA4CATA)
library(data4PCCAR)
library(plyr);library(dplyr)
library(corrplot)
library(ggplot2)
library(cowplot)
library(readr)
library(gridExtra)
library(grid)
library(readxl)
library(here)
library(psych)
# print versions pls
sessionInfo()




#**Data :**  cleaned data, pre processed and grouped into two different datasets - one for RSI measures and other for Air pollutants exposure measures from ABCD data.
#N = 8796


# load cleaned and prepped  RSI data
PROJ_DIR = '/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1'
DATA_DIR = 'data'
FIGS_DIR = "figures"
OUTP_DIR = "output"
df <- read.csv(paste(PROJ_DIR, DATA_DIR, 'data_qcd3.csv', sep = '/', collapse = NULL), header = TRUE, row.names = 1)

rni_cols <- colnames(df[, grep(pattern="dmri_rsirnigm.*\\change_score", colnames(df))])
ap_cols <- c("F1", "F2", "F3", "F4", "F5", "F6")
cov_cols <- c("sex",
              "interview_age",
              "ehi_y_ss_scoreb",
              "site_id_l",
              "mri_info_manufacturer",
              "physical_activity1_y",
              "stq_y_ss_weekday",
              "stq_y_ss_weekend",
              "reshist_addr1_proxrd",
              "reshist_addr1_popdensity",
              "reshist_addr1_urban_area",
              "nsc_p_ss_mean_3_items",
              "demo_comb_income_v2",
              "race_ethnicity"
)
fd_cols <- c("dmri_rsi_meanmotion", "dmri_rsi_meanmotion2")

temp_df <- df[,c(rni_cols, ap_cols, cov_cols, fd_cols)]

complete_df <- drop_na(temp_df)

#load cleaned and prepped  Air pollutants data
df_air <- complete_df[, ap_cols]
df_rsi <- complete_df[, rni_cols]
df_covariates <- complete_df[, cov_cols]


## Performing residualization 

###  MLRM for RSI data

# dependent variables
Group1.Y <- as.matrix(df_rsi)
dim(Group1.Y)


# independent variables/covariates
Group1.X <- df_covariates
dim(Group1.X)

Group1.X$dmri_rsi_meanmotion = (complete_df$dmri_rsi_meanmotion + complete_df$dmri_rsi_meanmotion2) / 2

# note:
# 1) check all variables- continous/factor
# 2) run str()
# 3) check for normality, homoscedascity,linearity,collinearity

# Multiple regression model
lm.Group1<- lm(Group1.Y ~ Group1.X$interview_age +
                 as.factor(Group1.X$race_ethnicity) +
                 Group1.X$physical_activity1_y +
                 (Group1.X$demo_comb_income_v2)+
                 Group1.X$stq_y_ss_weekday +
                 Group1.X$stq_y_ss_weekend +
                 Group1.X$nsc_p_ss_mean_3_items +
                 Group1.X$reshist_addr1_proxrd +
                 Group1.X$reshist_addr1_popdensity +
                 as.factor(Group1.X$reshist_addr1_urban_area) +
                 Group1.X$sex +
                 Group1.X$ehi_y_ss_scoreb +
                 as.factor(Group1.X$mri_info_manufacturer) + as.factor(Group1.X$site_id_l)+
                 Group1.X$dmri_rsi_meanmotion, na.action = na.omit) # data =)



# R.square -> how well the model explains the variation in the data which is not random
# Theorotical model performace is defined as R square

Group1_residuals <- data.frame()
Group1_residuals<- as.data.frame(lm.Group1$residuals)

#### Measure goodness of fit

#{r echo=TRUE, fig.height=10, fig.width=12}
# for one outcome variable
#plot(lm.Group1)

# for matrix of outcome variables
# loop the variables to obtain qq plot

for (i in seq_len(ncol(Group1_residuals))) {
  qqnorm(Group1_residuals[,i],  main=paste0("Q-Q plot for ",colnames(Group1_residuals)[i],sep=""))
  qqline(Group1_residuals[,i])
}

#hist.data.frame(as.data.frame(Group1_residuals))
# evaluation by predicton

f1 <- fitted(lm.Group1)
r1 <- residuals(lm.Group1)


for(i in seq_len(ncol(f1))) {
  plot(f1[,i],r1[,i] , main=paste0("fitted VS residual:   ",colnames(f1)[i],sep=""))
}





### MLRM for Air pollutants exposure data

#{r echo=TRUE}

#names(df_air)[8:10] <-  c("PM2.5", "NO2", "O3")
Group2.Y <- as.matrix(df_air)
corP_pollutants <- cor(df_air)
corK_pollutants<- cor(df_air, method = "kendall")
corSP_pollutants<- cor(df_air, method = "spearman")


lm.Group2<- lm(Group2.Y ~ Group1.X$interview_age +
                 as.factor(Group1.X$race_ethnicity) +
                 Group1.X$physical_activity1_y +
                 (Group1.X$demo_comb_income_v2)+
                 Group1.X$stq_y_ss_weekday +
                 Group1.X$stq_y_ss_weekend +
                 Group1.X$nsc_p_ss_mean_3_items +
                 Group1.X$reshist_addr1_proxrd +
                 Group1.X$reshist_addr1_popdensity +
                 as.factor(Group1.X$reshist_addr1_urban_area) +
                 Group1.X$sex +
                 Group1.X$ehi_y_ss_scoreb +
                 as.factor(Group1.X$mri_info_manufacturer) + as.factor(Group1.X$site_id_l)+
                 Group1.X$dmri_rsi_meanmotion, na.action = na.omit)

Group2_residuals<- data.frame()
Group2_residuals<-as.data.frame(lm.Group2$residuals) 






#### Measure goodness of fit

#{r echo=TRUE}

for (i in seq_len(ncol(Group2_residuals))) {
  
  qqnorm(Group2_residuals[,i],  main=paste0("Q-Q plot for ",colnames(Group2_residuals)[i],sep=""))
  qqline(Group2_residuals[,i])
}

#hist.data.frame(as.data.frame(Group2_residuals))

plot(fitted(lm.Group2), residuals(lm.Group2))
f <- fitted(lm.Group2)
r <- residuals(lm.Group2)

# add linear smoother 
for(i in seq_len(ncol(f))) {
  plot(f[,i],r[,i] , main=paste0("fitted VS residual:   ",colnames(f)[i],sep=""))
}



## Setting up colors for Analysis



# setting colors for RSI brain regions
#colors_brain <- c("pink","blue","yellow","deeppink","plum","skyblue","violet","darkgreen")
col4col_n0<- as.matrix(c("#f18c9c", "#f18d93", "#f18e8a", "#f18f7f", "#f28f71", "#f2905f", "#f0934d", "#e7984d", "#de9c4c", "#d79f4c", "#d0a24c", "#caa44c", "#c4a74c", "#bea94c", "#b8ab4c", "#b2ad4c", "#abaf4c", "#a4b14c", "#9db34c", "#94b54c", "#8bb74c", "#7fb94c", "#70bc4c", "#5bbe4c", "#4cc05a", "#4dbe6f", "#4ebe7d", "#4ebd87", "#4fbc90", "#4fbc97", "#50bb9d", "#50bba3", "#51baa8", "#51baac", "#51b9b1", "#52b9b5", "#52b8ba", "#53b8be", "#53b7c3", "#53b7c7", "#54b6cc", "#55b5d2", "#55b5d8", "#56b4df", "#57b2e8", "#61b0ee", "#74adee", "#83aaee", "#90a7ee", "#9ba4ee", "#a5a1ee", "#af9eee", "#b89bee", "#c198ee", "#ca94ee", "#d38fee", "#dc8bee", "#e785ee", "#ee81e9", "#ee83e0", "#ef84d7", "#ef86cf", "#ef87c8", "#ef88c1", "#f089ba", "#f089b3", "#f08aac", "#f08ba5"))

col4rsi<- col4col_n0


# setting colors for air pollutants
col4air <- c('#d8365d', '#8c7333', '#438432', '#37817b', '#3c7aad', '#ba3fc5')

col4column_rsi<- matrix(nrow = ncol(df_rsi), ncol = 1)
color.vec <- as.matrix(unique(colnames(df_rsi)))
for(i in seq_len(unique(colnames(df_rsi)))){
  col4column_rsi[which(color.vec == (unique(colnames(df_rsi)))[i])] <-col4rsi[i]
}

seq_along()


# colors set by site
library(RColorBrewer)
color_site <- brewer.pal(4,"PuOr")
color_site<- colorRampPalette(color_site)(21)
names(color_site) <- levels(unique(df_covariates$site))
colScale <- scale_colour_manual(name = "site",values = color_site)




# Color for rows
col4row  <- matrix(nrow = nrow(df_covariates), ncol = 1)

for (i in seq_len(unique(df_covariates$site))) {
  col4row[which(df_covariates$site == unique(df_covariates$site)[i])] <- color_site[i]
}


# Colors set by area
color_area <- brewer.pal(4,"PuOr")
color_area<- colorRampPalette(color_area)(3)
names(color_site) <- levels(unique(df_covariates$reshist_addr1_urban_area))
colScale <- scale_colour_manual(name = "area",values = color_site)

col4row_area  <- matrix(nrow = nrow(df_covariates), ncol = 1)

for (i in seq_len(unique(df_covariates$reshist_addr1_urban_area))) {
  col4row_area[which(df_covariates$reshist_addr1_urban_area == unique(df_covariates$reshist_addr1_urban_area)[i])] <- color_area[i]
}


col4row_sex <- df_covariates$sex
col4row_sex <- as.matrix(recode(col4row_sex,
                                "F" =  "red",
                                "M" =  "green",
))

# colors set by age
df_covariates$age<-cut(df_covariates$interview_age, breaks=c(0, 114, 120, 126,133), labels=c("9-9.5 yrs", "9.5-10 yrs", "10-10.5 yrs", "10.5-11 yrs"))

col4row_age <- df_covariates$age
col4row_age <- as.matrix(recode(col4row_age,
                                "9-9.5 yrs" =  "red",
                                "9.5-10 yrs" =  "plum",
                                "10-10.5 yrs" =  "mediumorchid3",
                                "10.5-11 yrs" =  "royalblue"))





## PLSC Analysis

### 1. Correlation plots

#{r fig.height= 5, fig.width=20, fig.align='center'}

#Input data for PLSC
data1 <- as.data.frame(Group1_residuals)
data2 <-as.data.frame(Group2_residuals)


# Compute the covariance matrix

XY.cor.pearson <- cor(data2,data1)


corrplot(XY.cor.pearson, method = "color", tl.cex = 1, tl.col = col4rsi, tl.pos='n',
         addCoef.col = "black", number.digits = 3, number.cex = .6,
         cl.pos = 'b', cl.cex = .7,title = "Pearson Correlation",
         addCoefasPercent = F, mar=c(0,0,1,0),
         col = colorRampPalette(c("darkred", "white","midnightblue"))(6))


#XY.cor.kendall <- cor(data1, data2, method = "kendall")
# corrplot(XY.cor.kendall, method = "color", tl.cex = 1, tl.col = col4rsi,
#          addCoef.col = "black", number.digits = 3, number.cex = .6, 
#          cl.pos = 'b', cl.cex = .7, title = "Kendall Correlation",
#          addCoefasPercent = F, mar=c(0,0,1,0),
#          col = colorRampPalette(c("darkred", "white","midnightblue"))(6)) 




### 2. Package details


tepPLS(data1, data2, center1 = TRUE, scale1 = "SS1", center2 = TRUE, scale2 = "SS1", DESIGN = NULL, make_design_nominal = TRUE, graphs = TRUE, k = 0)

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


# Applying PLSC function to the prepped data
pls.res <- tepPLS(data1,data2, DESIGN = data.design,  make_design_nominal = TRUE, graphs = FALSE)



### 3. Scree Plot

# The scree plot shows a weird pattern because the null hypothesis is that there is no correlation between the tables (Null = 0) 
# Hence eigenvalues greater than zero become significant. 

# The results of the permutation test gives us the eigenvalues.

# {r echo=TRUE}
# no.of eigenvalues
nL <- min(ncol(data1),ncol(data2))


# Applying permutation test to the input data for PLSC 
resPerm4PLSC <- perm4PLSC(data1, # First Data matrix 
                          data2, # Second Data matrix
                          permType = 'byColumns',
                          nIter = 10000 # How many iterations
)
print(resPerm4PLSC)

scree_df <- data.frame(row.names = c("f1", "f2", "f3", "f4", "f5", "f6"))
scree_df$eigens = pls.res$TExPosition.Data$eigs
scree_df$pEigens = resPerm4PLSC$pEigenvalues

library("writexl")
write.csv(scree_df,paste(PROJ_DIR, OUTP_DIR, paste("rni-scree_values_", resPerm4PLSC$pOmnibus, ".csv", sep = ""), sep = "/"))


# obtaining the scree plot
my.scree <-PlotScree(ev = pls.res$TExPosition.Data$eigs,
                     title = 'PLSC- Scree Plot',
                     p.ev = resPerm4PLSC$pEigenvalues,
                     plotKaiser = TRUE, 
                     color4Kaiser = ggplot2::alpha('darkorchid4', .5),
)


my.scree
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-PLSC_scree.png', sep="/"))
dev.off()


# additional permutation histogram plots

zeDim = 1
eigs <- resPerm4PLSC$permEigenvalues[,zeDim]
obs <- as.numeric(resPerm4PLSC$fixedEigenvalues[zeDim])
pH1 <- prettyHist(
  distribution = eigs, 
  observed = obs,
  xlim = c(0.002,0.018), # needs to be set by hand
  breaks = 30,
  border = "white",
  main = paste0("Permutation Test "),
  xlab = paste0("Inertia of samples
Latent dimension ",zeDim),
  ylab = "Number of samples",
  counts = F)
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_perm_hist.png', sep="/"))
dev.off()

zeDim = 2
eigs <- resPerm4PLSC$permEigenvalues[,zeDim]
obs <- as.numeric(resPerm4PLSC$fixedEigenvalues[zeDim])
pH2 <- prettyHist(
  distribution = eigs, 
  observed = obs,
  xlim = c(0.002,0.018), # needs to be set by hand
  breaks = 30,
  border = "white",
  main = paste0("Permutation Test "),
  xlab = paste0("Inertia of samples
Latent dimension ",zeDim),
  ylab = "Number of samples",
  counts = F)
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD2_perm_hist.png', sep="/"))
dev.off()


### 4. Looking at the First Pair of Latent variables by site 



# ploting the first latent variable of data1(X) and first latent variables of data2(Y). We are tryingto see if these two latent variables are similar or not.

# first pair of latent variables:

latvar.1 <- cbind(pls.res$TExPosition.Data$lx[,1],
                  pls.res$TExPosition.Data$ly[,1])
colnames(latvar.1) <- c("Lx 1", "Ly 1")

latvar.2 <- cbind(pls.res$TExPosition.Data$lx[,2],
                  pls.res$TExPosition.Data$ly[,2])
colnames(latvar.2) <- c("Lx 2", "Ly 2")

# compute means
lv.1.group <- getMeans(latvar.1,data.design.vec)
lv.2.group <- getMeans(latvar.2,data.design.vec)

col4Means <- as.matrix(color_site)
rownames(col4Means) <- rownames(lv.1.group)

# compute bootstrap - for confidence intervals
lv.1.group.boot <- Boot4Mean(latvar.1, data.design.vec)
colnames(lv.1.group.boot$BootCube) <- c("Lx 1", "Ly 1")

rownames(latvar.1) <- df_covariates$subjectkey

# plotiing the factor Maps
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
                              force =0.1,
                              alpha.points = 0.8)

plot1.meanCI <- MakeCIEllipses(lv.1.group.boot$BootCube[,c(1:2),], # get the first two components
                               col = col4Means[rownames(lv.1.group.boot$BootCube)],
                               names.of.factors = c("Lx 1", "Ly 1")
)

plot1 <- plot.lv1$zeMap_background + plot.lv1$zeMap_dots+ plot.lv1$zeMap_text + plot1.mean$zeMap_dots + plot1.mean$zeMap_text + plot1.meanCI 
plot1
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_factor_map.png', sep="/"))
dev.off()

# check for the outlier
outlier <- subset(latvar.1, rowSums(latvar.1 > 0.3) > 0)

indx <- which(rownames(latvar.1)== rownames(outlier))
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
                              force =0.1,
                              alpha.points = 0.8)

plot2.meanCI <- MakeCIEllipses(lv.2.group.boot$BootCube[,c(1:2),], # get the first two components
                               col = col4Means[rownames(lv.2.group.boot$BootCube)],
                               names.of.factors = c("Lx 2", "Ly 2")
)

plot2 <- plot.lv2$zeMap_background + plot.lv2$zeMap_dots+ plot.lv2$zeMap_text + plot2.mean$zeMap_dots + plot2.mean$zeMap_text + plot2.meanCI 
plot2
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD2_factor_map.png', sep="/"))
dev.off()

# check for the outlier
outlier <- subset(latvar.2, rowSums(latvar.2 > 0.3) > 0)

indx <- which(rownames(latvar.2)== rownames(outlier))
print(rownames(outlier))


### 5. Obtaining Column factor scores

# {r echo=TRUE}
### Column Factor scores of the 1st component of data1 representing RSI measures and data2 representing air pollutants

#Fi:column factor scores for data1(RSI measures) or Loadings of data1(RSI measures)
Fi<- pls.res$TExPosition.Data$fi

#Fi:column factor scores for data2(Air pollutants exposure) or Loadings of data2(air pollutants exposure)
Fj<- pls.res$TExPosition.Data$fj

# Generating loadings map of Fi
p.loadings <- createFactorMap(Fi,
                              axis1 = 1,
                              axis2 = 2,
                              display.points = TRUE,
                              display.labels = T,
                              col.points = col4column_rsi,
                              col.labels = col4column_rsi,
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


p.plot
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-region_loading_lv1xlv2.png', sep="/"))
dev.off()

# Generating loadings map of Fj
q.loadings <- createFactorMap(Fj,
                              axis1 = 1,
                              axis2 = 2,
                              display.points = TRUE,
                              display.labels = TRUE,
                              col.points = col4air,
                              col.labels = col4air,
                              title = "Loadings of Columns Air",
                              pch = 20,
                              cex = 3,
                              text.cex = 4,
)

label4map <- createxyLabels.gen(x_axis = 1, y_axis = 2,
                                lambda = pls.res$TExPosition.Data$eigs,
                                tau = pls.res$TExPosition.Data$t
)

q.plot <- q.loadings$zeMap + label4map 

q.plot
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-apsource_loading_lv1xlv2.png', sep="/"))
dev.off()



### 5. Column Loadings  

# These are the loadings obtained after performing SVD(R) on correlation matrix.


# {r echo=TRUE}

#generating bar plots for loadings of RSI data table
P.data1<- pls.res$TExPosition.Data$pdq$p


plot_P.data1 <- PrettyBarPlot2(
  bootratio = P.data1[,1], 
  threshold = 0, 
  ylim = NULL, 
  color4bar = gplots::col2hex(col4rsi),
  color4ns = "gray75", 
  plotnames = TRUE, 
  main = 'Loadings of RSI variables', 
  ylab = " P Loadings ",
  horizontal = TRUE)


plot_P.data1 
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_brain_loadings.png', sep="/"))
dev.off()

#generating bar plots for loadings of Air pollutants data table
Q.data2<- pls.res$TExPosition.Data$pdq$q


plot_Q.data2 <- PrettyBarPlot2(
  bootratio = Q.data2[,1], 
  threshold = 0, 
  ylim = NULL, 
  color4bar = gplots::col2hex(col4air),
  color4ns = "white", 
  plotnames = TRUE, 
  main = 'Loadings of Air pollution variables', 
  ylab = " Q Loadings ")



plot_Q.data2
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_apsource_loadings.png', sep="/"))
dev.off()

# saving the loading table into excel file.
air_loadings <- as.data.frame(Q.data2)
rni_loadings <- as.data.frame(P.data1)

#air_loadings <-cbind(" "=rownames(air_loadings), air_loadings)
#brain_loadings <-cbind(" "=rownames(brain_loadings), brain_loadings)
#library("writexl")
#write_xlsx(df_loadings,paste(PROJ_DIR, OUTP_DIR, "plsc_loadings_rni.csv", sep = "/"))


### 6. Inference Bootstrap


# The Bootstrap ratio barplot show that the contributions are significantly stable.
# {r echo=TRUE}
#Looking into what the resBootPLSC is giving us 
resBoot4PLSC <- Boot4PLSC(data1, # First Data matrix 
                          data2, # Second Data matrix
                          nIter = 10000, # How many iterations
                          Fi = pls.res$TExPosition.Data$fi,
                          Fj = pls.res$TExPosition.Data$fj,
                          nf2keep = 2,
                          critical.value = 2,
                          # To be implemented later
                          # has no effect currently
                          alphaLevel = 1)

resBoot4PLSC




# {r echo=TRUE}
BR.I <- resBoot4PLSC$bootRatios.i
BR.J <- resBoot4PLSC$bootRatios.j

# saving the bootstrap rations into excel

air_bootstrap <-  as.data.frame(BR.J)
rni_bootstrap <-  as.data.frame(BR.I)

air_res <- cbind.data.frame(air_loadings, air_bootstrap)
rni_res <- cbind.data.frame(rni_loadings, rni_bootstrap)


write.csv(air_res,paste(PROJ_DIR, 
                        OUTP_DIR, 
                        "rni-ap_source_Q.csv", 
                        sep = "/"), row.names = T)

write.csv(rni_res,paste(PROJ_DIR, 
                        OUTP_DIR, 
                        "rni-brain_P.csv", 
                        sep = "/"), row.names = T)


#bootstrap ratios for dimension 1
laDim = 1

ba001.BR1.I <- PrettyBarPlot2(BR.I[,laDim],
                              threshold = 2,
                              font.size = 4,
                              color4ns = "gray85", 
                              color4bar = col4rsi, # we need hex code
                              ylab = 'Bootstrap ratios'
                              #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
) + ggtitle(paste0('Latent dimension ', laDim), subtitle = 'Grey Matter Isotropic Intracellular Diffusion Saliences (Loadings)')

ba002.BR1.J <- PrettyBarPlot2(BR.J[,laDim],
                              threshold = 2,
                              font.size = 4,
                              color4ns = "gray85", 
                              color4bar = gplots::col2hex(col4air),
                              ylab = 'Bootstrap ratios'
                              #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
) + ggtitle(paste0('Latent dimension ', laDim), subtitle = 'Air Pollution Source Saliences (loadings)')

ba001.BR1.I
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_brain_saliences.png', sep="/"))
dev.off()
ba002.BR1.J 
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD1_apsource_saliences.png', sep="/"))
dev.off()




# {r echo=TRUE}
BR.I <- resBoot4PLSC$bootRatios.i
BR.J <- resBoot4PLSC$bootRatios.j


#bootstrap ratios for dimension 2
laDim = 2

ba001.BR1.I <- PrettyBarPlot2(BR.I[,laDim],
                              threshold = 2,
                              font.size = 4,
                              color4bar = col4rsi, # we need hex code
                              ylab = 'Bootstrap ratios'
                              #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
) + ggtitle(paste0('Latent dimension ', laDim), subtitle = 'Grey Matter Isotropic Diffusion Saliences (Loadings)')

ba002.BR1.J <- PrettyBarPlot2(BR.J[,laDim],
                              threshold = 2,
                              font.size = 4,
                              color4bar = gplots::col2hex(col4air),
                              ylab = 'Bootstrap ratios'
                              #ylim = c(1.2*min(BR[,laDim]), 1.2*max(BR[,laDim]))
) + ggtitle(paste0('Latent dimension ', laDim), subtitle = 'Air Pollution Source Saliences (loadings)')

ba001.BR1.I
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD2_brain_saliences.png', sep="/"))
dev.off()
ba002.BR1.J 
dev.copy(png, paste(PROJ_DIR, FIGS_DIR, 'rni-LD2_apsource_saliences.png', sep="/"))
dev.off()

