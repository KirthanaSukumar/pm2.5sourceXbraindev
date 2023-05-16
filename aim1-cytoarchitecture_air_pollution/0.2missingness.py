#!/usr/bin/env python
# coding: utf-8
import json
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from os.path import join, exists
from os import makedirs
from scipy.stats import pointbiserialr

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

sns.set(
    context='paper', 
    style='white', 
    palette=sns.husl_palette(
        n_colors=20, 
        h=0.01, 
        s=0.9, 
        l=0.65, 
        as_cmap=False
    )
)

if not exists(join(PROJ_DIR, OUTP_DIR)):
    makedirs(join(PROJ_DIR, OUTP_DIR))
else:
    pass

if not exists(join(PROJ_DIR, FIGS_DIR)):
    makedirs(join(PROJ_DIR, FIGS_DIR))
else:
    pass 

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data_qcd3.csv"), header=0, index_col=0)

rni_change_scores = df.filter(regex="dmri_rsirni.*change_score", axis=1).columns
rni_mean_change = df[rni_change_scores].dropna().mean(axis=1)
rni_mean_change.name = "dmri_rsirni_meanchange"
rnd_change_scores = df.filter(regex="dmri_rsirnd.*change_score", axis=1).columns
rnd_mean_change = df[rni_change_scores].dropna().mean(axis=1)
rnd_mean_change.name = "dmri_rsirnd_meanchange"
df = pd.concat([df, rni_mean_change, rnd_mean_change], axis=1)

numerical_vars = [
    "dmri_rsi_meanmotion",
    "dmri_rsi_meanmotion2",
    "dmri_rsirnd_meanchange",
    "dmri_rsirni_meanchange",  # MS
    "physical_activity1_y",  # MS
    "stq_y_ss_weekday",  # MS
    "stq_y_ss_weekend",  # MS
    "interview_age",
    "ehi_y_ss_scoreb",  # MS
    "reshist_addr1_proxrd",  # MS
    "reshist_addr1_popdensity",  # MS
    "nsc_p_ss_mean_3_items",  # MS
    "reshist_addr1_pm25",
    "F1",  # MS
    "F2",  # MS
    "F3",  # MS
    "F4",  # MS
    "F5", # MS
    "F6" # MS
]

categorical_vars = [ # i.e., to be dumbified
    "mri_info_manufacturer",  # MS
    "reshist_addr1_urban_area",  # MS
    "demo_comb_income_v2", # MS
    "demo_prnt_marital_v2",
    "demo_prnt_ed_v2",
    "race_ethnicity",  # MS
    "site_id_l",  # MS
]
# create dummies
all_dumbs = {}
for variable in categorical_vars:
    dumbies = pd.get_dummies(df[variable], prefix=variable, dummy_na=True)
    all_dumbs[variable] = list(dumbies.columns)
    dumbies[f'{variable}_nan'].replace({1:np.nan}, inplace=True)
    df = pd.concat([df, dumbies], axis=1)
# first, create response indicator matrix using model-destined variables
# variables we care about
model_vars = [
    "dmri_rsi_meanmotion",
    "dmri_rsi_meanmotion2",
    "dmri_rsirnd_meanchange",
    "dmri_rsirni_meanchange", # MS
    "physical_activity1_y",  # MS
    "stq_y_ss_weekday",  # MS
    "stq_y_ss_weekend",  # MS
    "interview_age",
    "ehi_y_ss_scoreb",  # MS
    "reshist_addr1_proxrd",  # MS
    "reshist_addr1_popdensity",  # MS
    "nsc_p_ss_mean_3_items",  # MS
    "reshist_addr1_pm25",
    "F1",  # MS
    "F2",  # MS
    "F3",  # MS
    "F4",  # MS
    "F5", # MS
    "F6" # MS
]

na_indicators = model_vars + [item for sublist in all_dumbs.values() for item in sublist if 'nan' in item]
model_vars += [item for sublist in all_dumbs.values() for item in sublist]
#print(model_vars)
response_indicator = df[na_indicators].isnull()
# then, calculate the % missingness
missingness = response_indicator.sum(axis=0) / len(df.index)

# calculate missingness due to imaging qc
with_2tmpts = response_indicator[response_indicator['dmri_rsi_meanmotion'] == 0]
with_2tmpts = with_2tmpts[with_2tmpts["dmri_rsi_meanmotion2"] == 0]
qc_missing = with_2tmpts['dmri_rsirnd_meanchange'].sum() / len(df.index)
missingness.at['dmri_qc_missingness'] = qc_missing
missingness.to_csv(join(PROJ_DIR, OUTP_DIR, 'proportion_missing.csv'))

# then check if missingness on each var is related to values of any other vars -- MAR if yes
stats = ['r', 'p']
columns = pd.MultiIndex.from_product([model_vars, stats])
missingness_corrs = pd.DataFrame(index=model_vars, columns=columns)
for variable in na_indicators:
    for variable2 in model_vars:
        if variable != variable2:
            no_nans = df[variable2].dropna().index
            #print(variable2, len(no_nans))
            r,p = pointbiserialr(response_indicator.loc[no_nans][variable], df.loc[no_nans][variable2])
            missingness_corrs.at[variable, (variable2, 'r')] = r
            missingness_corrs.at[variable, (variable2, 'p')] = p

missingness_corrs.to_csv(join(PROJ_DIR, OUTP_DIR, "missingness_correlations.csv"))
# how many patterns of missingness and what are they?
patterns = response_indicator[response_indicator.columns].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
pattern_counts = patterns.value_counts()
pattern_labels = []
for i in pattern_counts.index:
    # turn concatenated string back into boolean list
    pattern = [eval(x) for x in i.split(',')]
    #print(pattern)
    # yoink only the column names where pattern == True
    # these are the columns with missingness in the pattern
    the_missing_ones = response_indicator.columns[pattern]
    pattern_labels.append(', '.join(the_missing_ones))
fig, ax = plt.subplots(figsize=(20,10))
pattern_counts.plot.bar(ax=ax)
ax.set_xticklabels(range(0,len(pattern_labels)))
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'missingness_patterns.png'), dpi=400, bbox_inches="tight")
mapping = {}
for i in range(0, len(pattern_labels)):
    mapping[int(i)] = (str(pattern_labels[i]), int(pattern_counts.iloc[i]))
#print(mapping)
with open(join(PROJ_DIR, FIGS_DIR, 'missingness_patterns.json'), 'w') as f:
    # write the dictionary to the file in JSON format
    json.dump(mapping, f)

# use missingno?
fig = msno.matrix(df[na_indicators])
fig_copy = fig.get_figure()
fig_copy.savefig(join(PROJ_DIR, FIGS_DIR, 'missingno_matrix.png'), dpi=400, bbox_inches="tight")