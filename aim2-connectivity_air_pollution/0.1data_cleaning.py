#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from os.path import join

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)
all_subj = df.index

model_vars = [
    "mri_info_manufacturer",
    "dmri_rsi_meanmotion",
    "dmri_rsi_meanmotion2",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "race_ethnicity", 
    "rel_family_id",
    "sex",
    "interview_age",
    "ehi_y_ss_scoreb",
    "site_id_l",
    "interview_date",
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items"
    ]

mri_vars = list(df.filter(regex=".*_rsir.*.change_score", axis=1).columns)

model_vars  = model_vars + mri_vars

# modality-specific filtering via masks
# t1 quality for freesurfer ROI delineations
smri_mask1 = df['imgincl_t1w_include'] == 0
smri_mask2 = df['imgincl_t1w_include2'] == 0
smri_mask = smri_mask1 * smri_mask2

# dmri quality for RSI estimates
dmri_mask1 = df['imgincl_dmri_include'] == 0
dmri_mask2 = df['dmri_rsi_meanmotion'] < 2.
dmri_mask3 = df['imgincl_dmri_include2'] == 0
dmri_mask4 = df['dmri_rsi_meanmotion2'] < 2.
dmri_mask = dmri_mask1 * dmri_mask2 * dmri_mask3 * dmri_mask4

# and no incidental findings
findings1 = df['mrif_score'] >= 1.
findings2 = df['mrif_score'] <= 2.
findings3 = df['mrif_score2'] >= 1.
findings4 = df['mrif_score2'] <= 2.
findings_mask = findings1 * findings2 * findings3 * findings4


imaging_mask = smri_mask * dmri_mask * findings_mask

dmri_cols = df.filter(regex='dmri').columns
other_cols = set(df.columns) - set(dmri_cols)

# mask mri data
dmri_pass_subj = df.mask(imaging_mask).index
dmri_quality = df.loc[dmri_pass_subj]

other = df[other_cols]

masked = df.loc[dmri_pass_subj]
complete_cases = df[model_vars].dropna(how='any').index

complete_df = df.loc[complete_cases]

# I want to compare
# 1. the full dataset (i.e., regardless of missingness, quality, etc.)
# 2. dataset filtered for MRI quality
# 3. complete case data


quality_df = pd.concat([other, dmri_quality], axis=1)

quality_df.to_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"))

demographics = ["demo_prnt_marital_v2",
                "demo_prnt_ed_v2",
                "demo_comb_income_v2",
                "race_ethnicity",
                "site_id_l",
                "sex", 
                "mri_info_manufacturer"
               ]

mri_qc = [
    "imgincl_dmri_include",# baseline
    "imgincl_t1w_include",# baseline
    "mrif_score",# baseline
    "dmri_rsi_meanmotion", # baseline
    "imgincl_dmri_include2", # year 2 follow-up
    "imgincl_t1w_include2",# year 2 follow-up
    "mrif_score2",# year 2 follow-up
    "dmri_rsi_meanmotion", # year 2 follow-up
    "interview_age", # baseline
    "interview_date" # baseline
]


demo_and_qc = demographics + mri_qc

demo_df = df[demo_and_qc]

#qc_df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data_qcd.csv'), 
#                 header=0, 
#                 index_col='subjectkey')
#qc_ppts = qc_df.dropna(how='all').index
#qc_df = None

total_N = len(demo_df.index)

dmri_quality = demo_df.loc[dmri_quality.dropna().index]

table = pd.DataFrame(index=['N', 
                            'Age_mean_base',
                            'Age_sdev_base',
                            'Age_mean_2yfu',
                            'Age_sdev_2yfu',
                            'Sex_M', 
                            'Sex_F', 
                            'RE_Black',
                            'RE_White',
                            'RE_Hispanic',
                            'RE_AsianOther',
                            'Income_gt100k', 
                            'Income_50to100k', 
                            'Income_lt50k',
                            'Income_dkrefuse',
                            'Marital_Married',
                            'Marital_Widowed',
                            'Marital_Divorced',
                            'Marital_Separated',
                            'Marital_Never',
                            'Marital_Refused',
                            'Education_uptoHSGED',
                            'Education_SomeColAA',
                            'Education_Bachelors',
                            'Education_Graduate',
                            'Education_Refused',
                            'MRI_Siemens', 
                            'MRI_GE', 
                            'MRI_Philips'], 
                     columns=['whole_sample', 'dmri_pass', 'complete_case'])

table.at['N', 'whole_sample'] = len(all_subj)
table.at['N', 'with_dmri'] = len(dmri_pass_subj)
table.at['N', 'complete_case'] = len(complete_cases)

table.at['Age_mean_base', 'whole_sample'] = np.mean(df['interview_age'])
table.at['Age_mean_base', 'with_dmri'] = np.mean(df.loc[dmri_pass_subj]['interview_age'])
table.at['Age_mean_base', 'complete_case'] = np.mean(df.loc[complete_cases]['interview_age'])

table.at['Age_sdev_base', 'whole_sample'] = np.std(demo_df['interview_age'])
table.at['Age_sdev_base', 'with_dmri'] = np.std(dmri_quality['interview_age'])
table.at['Age_sdev_base', 'complete_case'] = np.std(complete_df['interview_age'])

table.at['Sex_M', 'whole_sample'] = len(demo_df[demo_df['sex'] == 'M'].index)
table.at['Sex_M', 'with_dmri'] = len(dmri_quality[dmri_quality['sex'] == 'M'].index)
table.at['Sex_M', 'complete_case'] = len(df[df['sex'] == 'M'].index)

table.at['Sex_F', 'whole_sample'] = len(demo_df[demo_df['sex'] == 'F'].index)
table.at['Sex_F', 'with_dmri'] = len(dmri_quality[dmri_quality['sex'] == 'F'].index)
table.at['Sex_F', 'complete_case'] = len(complete_df[complete_df['sex'] == 'F'].index)


table.at['RE_White', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity'] == 1.].index)
table.at['RE_White', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity'] == 1.].index)
table.at['RE_White', 
         'complete_case'] = len(complete_df[complete_df['race_ethnicity'] == 1.].index)

table.at['RE_Black', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity'] == 2.].index)
table.at['RE_Black', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity'] == 2.].index)
table.at['RE_Black', 
         'complete_case'] = len(complete_df[complete_df['race_ethnicity'] == 2.].index)

table.at['RE_Hispanic', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity'] == 3.].index)
table.at['RE_Hispanic', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity'] == 3.].index)
table.at['RE_Hispanic', 
         'complete_case'] = len(complete_df[complete_df['race_ethnicity'] == 3.].index)

table.at['RE_AsianOther', 
         'whole_sample'] = len(demo_df[demo_df['race_ethnicity'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'with_dmri'] = len(dmri_quality[dmri_quality['race_ethnicity'].between(4.,5.,inclusive='both')].index)
table.at['RE_AsianOther', 
         'complete_case'] = len(complete_df[complete_df['race_ethnicity'].between(4.,5.,inclusive='both')].index)


table.at['Income_gt100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2'].between(9.,10., inclusive='both')].index)
table.at['Income_gt100k', 
         'complete_case'] = len(complete_df[complete_df['demo_comb_income_v2'].between(9.,10., inclusive='both')].index)

table.at['Income_50to100k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2'] <= 6.].index)

table.at['Income_dkrefuse', 
         'whole_sample'] = len(demo_df[demo_df['demo_comb_income_v2'] >= 777.].index)

table.at['MRI_Siemens', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer'] == "SIEMENS"].index)
table.at['MRI_GE', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer'] == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_Philips', 
         'whole_sample'] = len(demo_df[demo_df['mri_info_manufacturer'] == "Philips Medical Systems"].index)

table.at['Marital_Married', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 1.])
table.at['Marital_Widowed', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 2.])
table.at['Marital_Divorced', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 3.])
table.at['Marital_Separated', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 4.])
table.at['Marital_Never', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 5.])
table.at['Marital_Refused', 
         'whole_sample'] = len(demo_df[demo_df["demo_prnt_marital_v2"] == 777.])

table.at['Education_uptoHSGED', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2'] == 18])
table.at['Education_Graduate', 
         'whole_sample'] = len(demo_df[demo_df['demo_prnt_ed_v2'].between(19,22, 
                                                                                                inclusive='both')])

table.at['Income_50to100k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2'].between(7., 8., inclusive='both')].index)
table.at['Income_50to100k', 
         'complete_case'] = len(complete_df[complete_df['demo_comb_income_v2'].between(7., 8., inclusive='both')].index)

table.at['Income_lt50k', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2'] <= 6.].index)
table.at['Income_lt50k', 
         'complete_case'] = len(complete_df[complete_df['demo_comb_income_v2'] <= 6.].index)

table.at['Income_dkrefuse', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_comb_income_v2'] >= 777.].index)
table.at['Income_dkrefuse', 
         'complete_case'] = len(complete_df[complete_df['demo_comb_income_v2'] >= 777.].index)

table.at['MRI_Siemens', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer'] == "SIEMENS"].index)
table.at['MRI_Siemens', 
         'complete_case'] = len(complete_df[complete_df['mri_info_manufacturer'] == "SIEMENS"].index)
table.at['MRI_GE', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer'] == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_GE', 
         'complete_case'] = len(complete_df[complete_df['mri_info_manufacturer']  == "GE MEDICAL SYSTEMS"].index)
table.at['MRI_Philips', 
         'with_dmri'] = len(dmri_quality[dmri_quality['mri_info_manufacturer'] == "Philips Medical Systems"].index)
table.at['MRI_Philips', 
         'complete_case'] = len(complete_df[complete_df['mri_info_manufacturer'] == "Philips Medical Systems"].index)

table.at['Marital_Married', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 1.])
table.at['Marital_Married', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 1.])
table.at['Marital_Widowed', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 2.])
table.at['Marital_Widowed', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 2.])
table.at['Marital_Divorced', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 3.])
table.at['Marital_Divorced', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 3.])
table.at['Marital_Separated', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 4.])
table.at['Marital_Separated', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 4.])
table.at['Marital_Never', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 5.])
table.at['Marital_Never', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 5.])
table.at['Marital_Refused', 
         'with_dmri'] = len(dmri_quality[dmri_quality["demo_prnt_marital_v2"] == 777.])
table.at['Marital_Refused', 
         'complete_case'] = len(complete_df[complete_df["demo_prnt_marital_v2"] == 777.])

table.at['Education_uptoHSGED', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2'] == 18])
table.at['Education_Graduate', 
         'with_dmri'] = len(dmri_quality[dmri_quality['demo_prnt_ed_v2'].between(19,22, 
                                                                                                inclusive='both')])
table.at['Education_uptoHSGED', 
         'complete_case'] = len(complete_df[complete_df['demo_prnt_ed_v2'].between(0,14, 
                                                                                                inclusive='both')])
table.at['Education_SomeColAA', 
         'complete_case'] = len(complete_df[complete_df['demo_prnt_ed_v2'].between(15,17, 
                                                                                                inclusive='both')])
table.at['Education_Bachelors', 
         'complete_case'] = len(complete_df[complete_df['demo_prnt_ed_v2'] == 18])
table.at['Education_Graduate', 
         'complete_case'] = len(complete_df[complete_df['demo_prnt_ed_v2'].between(19,22, 
                                                                                                inclusive='both')])
table.to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))