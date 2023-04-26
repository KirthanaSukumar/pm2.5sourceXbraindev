#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns

from os.path import join, isdir
from os import makedirs

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_csv(join(PROJ_DIR, DATA_DIR, "data.csv"), index_col=0, header=0)
all_subj = df.index

df['interview_date'] = pd.to_datetime(df['interview_date'])
df['interview_date2'] = pd.to_datetime(df['interview_date2'])
# QC filtering - censoring all ppts whose 2-year follow-up visit was after covid
# bc a global pandemic is a pretty serious confounder LOL
before_covid = df[df['interview_date2'] < '2020-3-1'].index
# also calculating demographics, etc. for all those with a valid address
valid_address = df[df['reshist_addr1_valid'] == 1].index

model_vars = [
    "mri_info_manufacturer",
    "rsfmri_c_ngd_ntpoints",
    "rsfmri_c_ngd_ntpoints2",
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
    'reshist_addr1_valid',
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
    "F6"
    ]

mri_vars = list(df.filter(regex=".*rsfmri.*.change_score", axis=1).columns)

model_vars  = model_vars + mri_vars

# modality-specific filtering via masks
# t1 quality for freesurfer ROI delineations
smri_mask1 = df['imgincl_t1w_include'] == 0
smri_mask2 = df['imgincl_t1w_include2'] == 0
smri_mask = smri_mask1 * smri_mask2

# rsfmri quality for FC estimates
rsfmri_mask1 = df['imgincl_rsfmri_include'] == 0
rsfmri_mask2 = df['rsfmri_c_ngd_ntpoints'] < 2.
rsfmri_mask3 = df['imgincl_rsfmri_include2'] == 0
rsfmri_mask4 = df['rsfmri_c_ngd_ntpoints2'] < 2.
rsfmri_mask = rsfmri_mask1 * rsfmri_mask2 * rsfmri_mask3 * rsfmri_mask4

# and no incidental findings
findings1 = df['mrif_score'] >= 1.
findings2 = df['mrif_score'] <= 2.
findings3 = df['mrif_score2'] >= 1.
findings4 = df['mrif_score2'] <= 2.
findings_mask = findings1 * findings2 * findings3 * findings4


imaging_mask = smri_mask * rsfmri_mask * findings_mask

rsfmri_cols = df.filter(regex='rsfmri').columns

# mask mri data
rsfmri_pass_subj = df[rsfmri_cols].mask(imaging_mask).dropna().index
rsfmri_quality = df.loc[rsfmri_pass_subj]


# I want to compare
# 1. the full dataset (i.e., regardless of missingness, quality, etc.)
# 2. dataset filtered for MRI quality
# 3. complete case data


quality_df = rsfmri_quality[rsfmri_quality['interview_date2'] < '2020-3-1']

quality_df.to_csv(join(PROJ_DIR, DATA_DIR, "data_qcd.csv"))

demographics = ["demo_prnt_marital_v2",
                "demo_prnt_ed_v2",
                "demo_comb_income_v2",
                "race_ethnicity",
                "site_id_l",
                "sex", 
                "mri_info_manufacturer",
                "reshist_addr1_valid"
               ]

mri_qc = [
    "imgincl_rsfmri_include",# baseline
    "imgincl_t1w_include",# baseline
    "mrif_score",# baseline
    "rsfmri_c_ngd_ntpoints", # baseline
    "imgincl_rsfmri_include2", # year 2 follow-up
    "imgincl_t1w_include2",# year 2 follow-up
    "mrif_score2",# year 2 follow-up
    "rsfmri_c_ngd_ntpoints2", # year 2 follow-up
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

rsfmri_quality = demo_df.loc[rsfmri_pass_subj]
pre_covid_df = df.loc[before_covid]
pre_covid_good_address = pre_covid_df[pre_covid_df['reshist_addr1_valid'] == 1]
rest_good = df.loc[rsfmri_pass_subj]
rest_good = rest_good[rest_good['interview_date2'] < '2020-3-1']

col_to_df = {
    'whole_sample': df, 
    'pre_covid': pre_covid_df,
    'with_address': pre_covid_good_address,
    'with_rsfmri': rest_good,
    'complete_cases': quality_df.dropna(how='any')
}

table = pd.DataFrame(index=['N', 
                            'Age_mean_base',
                            'Age_sdev_base',
                            'Age_Missing',
                            'Sex_M', 
                            'Sex_F', 
                            'Sex_Missing',
                            'RE_Black',
                            'RE_White',
                            'RE_Hispanic',
                            'RE_AsianOther',
                            'RE_Missing',
                            'RE_Refuse',
                            'Income_gt100k', 
                            'Income_50to100k', 
                            'Income_lt50k',
                            'Income_dkrefuse',
                            'Income_Missing',
                            'Marital_Married',
                            'Marital_Widowed',
                            'Marital_Divorced',
                            'Marital_Separated',
                            'Marital_Never',
                            'Marital_Refused',
                            'Marital_Missing',
                            'Education_uptoHSGED',
                            'Education_SomeColAA',
                            'Education_Bachelors',
                            'Education_Graduate',
                            'Education_Refused',
                            'Education_Missing',
                            'MRI_Siemens', 
                            'MRI_GE', 
                            'MRI_Philips',
                            'MRI_Missing',
                            "F1_mean",
                            "F1_sdev",
                            "F2_mean",
                            "F2_sdev",
                            "F3_mean",
                            "F3_sdev",
                            "F4_mean",
                            "F4_sdev",
                            "F5_mean",
                            "F5_sdev",
                            "F6_mean",
                            "F6_sdev",
                            "PM2.5_mean",
                            "PM2.5_sdev",], 
                     columns=list(col_to_df.keys()))

for subset in col_to_df.keys():
    #print(subset, type(col_to_df[subset]))
    temp_df = col_to_df[subset]
    table.at['N', subset] = len(temp_df.index)
    table.at['Age_mean_base', subset] = np.mean(temp_df['interview_age'])
    table.at['Age_sdev_base', subset] = np.std(temp_df['interview_age'])
    # PM2.5 factors
    table.at['F1_mean', subset] = np.mean(temp_df['F1'])
    table.at['F1_sdev', subset] = np.std(temp_df['F1'])

    table.at['F2_mean', subset] = np.mean(temp_df['F2'])
    table.at['F2_sdev', subset] = np.std(temp_df['F2'])

    table.at['F3_mean', subset] = np.mean(temp_df['F3'])
    table.at['F3_sdev', subset] = np.std(temp_df['F3'])

    table.at['F4_mean', subset] = np.mean(temp_df['F4'])
    table.at['F4_sdev', subset] = np.std(temp_df['F4'])

    table.at['F5_mean', subset] = np.mean(temp_df['F5'])
    table.at['F5_sdev', subset] = np.std(temp_df['F5'])

    table.at['F6_mean', subset] = np.mean(temp_df['F6'])
    table.at['F6_sdev', subset] = np.std(temp_df['F6'])

    table.at['PM2.5_mean', subset] = np.mean(temp_df['reshist_addr1_pm25'])
    table.at['PM2.5_sdev', subset] = np.std(temp_df['reshist_addr1_pm25'])

    # demographics
    table.at['Age_Missing', subset] = temp_df['interview_age'].isna().sum()
    table.at['Sex_M', subset] = len(temp_df[temp_df['sex'] == 'M'].index)
    table.at['Sex_F', subset] = len(temp_df[temp_df['sex'] == 'F'].index)
    table.at['Sex_Missing', subset] = temp_df['sex'].isna().sum()
    table.at['RE_White',
             subset] = len(temp_df[temp_df['race_ethnicity'] == 1.].index)
    table.at['RE_Black',
             subset] = len(temp_df[temp_df['race_ethnicity'] == 2.].index)
    table.at['RE_Hispanic',
             subset] = len(temp_df[temp_df['race_ethnicity'] == 3.].index)
    table.at['RE_AsianOther',
             subset] = len(temp_df[temp_df['race_ethnicity'].between(4.,5.,inclusive='both')].index)
    table.at['RE_Refuse',
             subset] = len(temp_df[temp_df['race_ethnicity'] == 777.].index)
    table.at['RE_Missing',
             subset] = temp_df['race_ethnicity'].isna().sum() + len(temp_df[temp_df['race_ethnicity'] == 999.].index)
    table.at['Income_gt100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'].between(9.,10., inclusive='both')].index)
    table.at['Income_50to100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'].between(7., 8., inclusive='both')].index)
    table.at['Income_lt50k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'] <= 6.].index)
    table.at['Income_dkrefuse', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2'] == 777.].index)
    table.at['Income_Missing', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2'] == 999.].index) + temp_df['demo_comb_income_v2'].isna().sum()
    table.at['MRI_Siemens', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "SIEMENS"].index)
    table.at['MRI_GE', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "GE MEDICAL SYSTEMS"].index)
    table.at['MRI_Philips', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "Philips Medical Systems"].index)
    table.at['MRI_Missing', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'].isna()].index)
    table.at['Marital_Married', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 1.])
    table.at['Marital_Widowed', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 2.])
    table.at['Marital_Divorced', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 3.])
    table.at['Marital_Separated', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 4.])
    table.at['Marital_Never', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 5.])
    table.at['Marital_Refused', 
            subset] = len(temp_df[temp_df["demo_prnt_marital_v2"] == 777.])
    table.at['Marital_Missing', 
            subset] = temp_df["demo_prnt_marital_v2"].isna().sum() + len(temp_df[temp_df["demo_prnt_marital_v2"] == 999.])
    table.at['Education_uptoHSGED', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2'].between(0,14,inclusive='both')])
    table.at['Education_SomeColAA', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2'].between(15,17, inclusive='both')])
    table.at['Education_Bachelors', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2'] == 18])
    table.at['Education_Graduate', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2'].between(19,22, inclusive='both')])
    table.at['Education_Refused', 
            subset] = len(temp_df[temp_df['demo_prnt_ed_v2'] == 777.])
    table.at['Education_Missing', 
            subset] = temp_df['demo_prnt_ed_v2'].isna().sum() + len(temp_df[temp_df['demo_prnt_ed_v2'] == 999.])

if not isdir(join(PROJ_DIR, OUTP_DIR)):
    makedirs(join(PROJ_DIR, OUTP_DIR))

table.to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))