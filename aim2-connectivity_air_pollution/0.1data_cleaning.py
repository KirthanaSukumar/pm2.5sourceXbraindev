#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import abcdWrangler as abcdw

from os.path import join, isdir
from os import makedirs

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

DEMO_VARS = [
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items",
    "ehi_y_ss_scoreb",
    #'reshist_addr1_Lnight_exi',
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    "site_id_l",
    'rel_family_id',
]

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data.pkl"))

df['interview_date'] = pd.to_datetime(df['interview_date'], format="%m/%d/%Y")
address = df['reshist_addr1_urban_area'].dropna().index.get_level_values(0).unique()
# scanner manufacturer & pre-covid only matters for change scores
# so we only need the ppt IDs 
siemens = df[df['mri_info_manufacturer'] == 'SIEMENS'].index.get_level_values(0).unique()
# QC filtering - censoring all ppts whose 2-year follow-up visit was after covid
# bc a global pandemic is a pretty serious confounder LOL
pre_covid = df[df["interview_date"] < '2020-03-01'].xs('2_year_follow_up_y_arm_1', level=1).index.get_level_values(0).unique()
change_score_eligible = list(set(siemens) & set(pre_covid) & set(address))

# returns index of ppts who meet inclusion criteria
good_fmri = abcdw.fmri_qc(df, ntpoints=750, motion_thresh=0.5)

good_fmri_base = [i[0] for i in good_fmri if i[1] == 'baseline_year_1_arm_1']
good_fmri_base = list(set(good_fmri_base) & set(address))
complete_base = df.loc[good_fmri_base][DEMO_VARS].dropna().index.get_level_values(0).unique()

siemens = list(set(siemens) & set(complete_base))
pre_covid = list(set(pre_covid) & set(siemens))
good_fmri_y2fu = [i[0] for i in good_fmri if i[1] == '2_year_follow_up_y_arm_1']

good_fmri_delta = list(set(good_fmri_y2fu) & set(pre_covid))

all_ppts = df.index.get_level_values(0).unique()
sample_size = pd.DataFrame(
    columns=[
        'keep', 
        'drop'
    ],
    index=[
        'ABCD Study',
        'Address',
        'fMRI QC base',
        'base complete',
        'SIEMENS',
        'Pre-COVID',
        'delta complete',
    ]
)


sample_size.at['ABCD Study', 'keep'] = len(all_ppts)
sample_size.at['ABCD Study', 'drop'] = 0
sample_size.at['Address', 'keep'] = len(address)
sample_size.at['Address', 'drop'] = len(all_ppts) - len(address)

# imaging quality control at baselien
sample_size.at['fMRI QC base', 'keep'] = len(good_fmri_base)
sample_size.at['fMRI QC base', 'drop'] = len(address) - len(good_fmri_base)
# complete case data baseline
sample_size.at['base complete', 'keep'] = len(complete_base)
sample_size.at['base complete', 'drop'] = len(good_fmri_base) - len(complete_base)

sample_size.at['SIEMENS', 'keep'] = len(siemens)
sample_size.at['SIEMENS', 'drop'] = len(complete_base) - len(siemens)

sample_size.at['Pre-COVID', 'keep'] = len(pre_covid)
sample_size.at['Pre-COVID', 'drop'] = len(siemens) - len(pre_covid)

sample_size.at['delta complete', 'keep'] = len(good_fmri_delta)
sample_size.at['delta complete', 'drop'] = len(pre_covid) - len(good_fmri_delta)
# complete case data for change scores

sample_size.to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_size_qc.csv'))


col_to_df = {
    'ABCD Study': all_ppts,
    'Address': address,
    'fMRI QC base': good_fmri_base,
    'base complete': list(set(complete_base)),
    'SIEMENS': siemens,
    'Pre-COVID': pre_covid,
    'delta complete': good_fmri_delta
    }

ppts = pd.DataFrame(
    index=all_ppts,
    columns=[
        'ABCD Study',
        'Address',
        'fMRI QC base',
        'base complete',
        'SIEMENS',
        'Pre-COVID',
        'delta complete',
    ]
)

for ppt in all_ppts:
    for key in col_to_df.keys():
        if ppt in col_to_df[key]:
            ppts.at[ppt, key] = 1

ppts.to_pickle(join(PROJ_DIR, OUTP_DIR, 'ppts_qc.pkl'))

model_vars = [
    "mri_info_manufacturer",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "race_ethnicity", 
    "rel_family_id",
    "sex",
    "interview_age",
    "ehi_y_ss_scoreb",
    #"site_id_l",
    "interview_date",
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items",
    "reshist_addr1_pm25",
    "F1", 
    "F2", 
    "F3", 
    "F4", 
    "F5", 
    "F6",
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    ]

base_df = df.loc[complete_base].xs('baseline_year_1_arm_1', level=1)
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_base.pkl"))

delta_df = df.loc[good_fmri_delta]
base_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_delta.pkl"))

table = pd.DataFrame(
    index=[
       "N"
    ], 
    columns=list(col_to_df.keys())
)

vars = [
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    'reshist_addr1_popdensity',
    'ehi_y_ss_scoreb', 
    'reshist_addr1_proxrd',
    'reshist_addr1_urban_area', 
    'nsc_p_ss_mean_3_items',
    "reshist_addr1_pm252016aa",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "F1", 
    "F2", 
    "F3", 
    "F4", 
    "F5", 
    "F6"
]


for subset in col_to_df.keys():
    #print(subset, type(col_to_df[subset]))
    ppts = col_to_df[subset]
    temp_df = df.loc[ppts].xs('baseline_year_1_arm_1', level=1)
    table.at['N', subset] = len(temp_df.index)
    
    for col in vars:
        table.at[f'{col}-missing', subset] = temp_df[col].isna().sum()
        if len(temp_df[col].unique()) < 6:
            counts = temp_df[col].value_counts()
            for level in counts.index:
                table.at[f'{col}-{level}',subset] = counts[level]
        else:
            table.at[f'{col}-mean',subset] = temp_df[col].mean()
            table.at[f'{col}-sdev',subset] = temp_df[col].std()

table.dropna(how='all').to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))