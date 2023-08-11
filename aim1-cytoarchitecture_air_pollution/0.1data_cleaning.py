#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from os.path import join, exists
from os import makedirs

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

if not exists(join(PROJ_DIR, OUTP_DIR)):
    makedirs(join(PROJ_DIR, OUTP_DIR))
else:
    pass

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data.pkl"))
print(df.columns)

df['interview_date'] = pd.to_datetime(df['interview_date'])
df['interview_date2'] = pd.to_datetime(df['interview_date2'])
# QC filtering - censoring all ppts whose 2-year follow-up visit was after covid
# bc a global pandemic is a pretty serious confounder LOL
before_covid = df[df['interview_date2'] < '2020-3-1'].index
# also calculating demographics, etc. for all those with a valid address
valid_address = df[df['reshist_addr1_valid'] == 1].index

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

mri_vars = list(df.filter(regex=".*_rsir.*.change_score", axis=1).columns)

model_vars  = model_vars + mri_vars

# modality-specific filtering via masks
# t1 quality for freesurfer ROI delineations
# 1=include, 0=exclude
smri_mask1 = df['imgincl_t1w_include'] == 1
smri_mask2 = df['imgincl_t1w_include2'] == 1
smri_mask = smri_mask1 * smri_mask2


# dmri quality for RSI estimates
# 1=include, 0=exclude
# head motion greater than 2mm FD on average = exclude
dmri_mask1 = df['imgincl_dmri_include'] == 1
dmri_mask2 = df['dmri_rsi_meanmotion'] < 2.
dmri_mask3 = df['imgincl_dmri_include2'] == 1
dmri_mask4 = df['dmri_rsi_meanmotion2'] < 2.
dmri_mask = dmri_mask1 * dmri_mask2 * dmri_mask3 * dmri_mask4


# and no incidental findings
# if true, then exclude
# mrif_score 3 or 4 recommended for exclusion?
# mrif_score 2 is normal anatomical variation
# mrif_score 1 is no findings
# mrif_score 0 means quality is too poor for rad read

findings1 = df['mrif_score'].between(1,2, inclusive='both')
#findings2 = df['mrif_score'] == 2
findings3 = df['mrif_score2'].between(1,2, inclusive='both')#
#findings4 = df['mrif_score2'] == 2
findings_mask = findings1 * findings3 #* findings3 * findings4

imaging_mask = smri_mask * dmri_mask * findings_mask

qc_fails = np.invert(imaging_mask)

dmri_cols = df.filter(regex='dmri.*change_score').columns

# mask mri data
# Replace values where the condition is True.
df[dmri_cols] = df[dmri_cols].mask(qc_fails)
dmri_pass_subj = qc_fails[qc_fails == False].index
print(len(dmri_pass_subj))

temp_df = df.loc[dmri_pass_subj][model_vars]
complete_cases = temp_df.dropna(how='any', axis=0).index
temp_df = None

complete_df = df.loc[complete_cases]

# I want to compare
# 1. the full dataset (i.e., regardless of missingness, quality, etc.)
# 2. dataset filtered for MRI quality
# 3. complete case data


quality_df = df[df['interview_date2'] < '2020-3-1']
print(np.sum(quality_df[dmri_cols[0]].isnull()))

#print(quality_df.index)
print("n dupes:", np.sum(quality_df.index.duplicated()), "out of", len(quality_df.index))
#print(quality_df.dtypes)
quality_df.replace({999.: np.nan, 777.: np.nan}, inplace=True)
quality_df.to_csv(join(PROJ_DIR, DATA_DIR, "data_qcd4.csv"), encoding='utf-8')
quality_df.to_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd.pkl"))


#qc_df = pd.read_csv(join(PROJ_DIR, DATA_DIR, 'data_qcd.csv'), 
#                 header=0, 
#                 index_col='subjectkey')
#qc_ppts = qc_df.dropna(how='all').index
#qc_df = None

pre_covid_df = df.loc[before_covid]
pre_covid_good_address = pre_covid_df[pre_covid_df['reshist_addr1_valid'] == 1]
dmri_good = df.loc[dmri_pass_subj]
dmri_good = dmri_good[dmri_good['interview_date2'] < '2020-3-1']

col_to_df = {
    'whole_sample': df, 
    'pre_covid': pre_covid_df,
    'with_address': pre_covid_good_address,
    'with_dmri': dmri_good,
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
    table.at['Age_Missing', subset] = temp_df['interview_age'].isnull().sum()
    table.at['Sex_M', subset] = len(temp_df[temp_df['sex'] == 'M'].index)
    table.at['Sex_F', subset] = len(temp_df[temp_df['sex'] == 'F'].index)
    table.at['Sex_Missing', subset] = temp_df['sex'].isnull().sum()
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
             subset] = temp_df['race_ethnicity'].isnull().sum() + len(temp_df[temp_df['race_ethnicity'] == 999.].index)
    table.at['Income_gt100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'].between(9.,10., inclusive='both')].index)
    table.at['Income_50to100k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'].between(7., 8., inclusive='both')].index)
    table.at['Income_lt50k', 
         subset] = len(temp_df[temp_df['demo_comb_income_v2'] <= 6.].index)
    table.at['Income_dkrefuse', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2'] == 777.].index)
    table.at['Income_Missing', 
            subset] = len(temp_df[temp_df['demo_comb_income_v2'] == 999.].index) + temp_df['demo_comb_income_v2'].isnull().sum()
    table.at['MRI_Siemens', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "SIEMENS"].index)
    table.at['MRI_GE', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "GE MEDICAL SYSTEMS"].index)
    table.at['MRI_Philips', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'] == "Philips Medical Systems"].index)
    table.at['MRI_Missing', 
            subset] = len(temp_df[temp_df['mri_info_manufacturer'].isnull()].index)
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
            subset] = temp_df["demo_prnt_marital_v2"].isnull().sum() + len(temp_df[temp_df["demo_prnt_marital_v2"] == 999.])
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
            subset] = temp_df['demo_prnt_ed_v2'].isnull().sum() + len(temp_df[temp_df['demo_prnt_ed_v2'] == 999.])


table.to_csv(join(PROJ_DIR, OUTP_DIR, 'demographics.csv'))