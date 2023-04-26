# importing the tools we'll use throughout the rest of the script
# sys is system tools, should already be installed
import sys
# enlighten gives you a progress bar, but it's optional
import enlighten
# pandas is a dataframe-managing library and it's the absolute coolest
import pandas as pd
import numpy as np
# os is more system tools, should also already be installed
# we're importing tools for verifying and manipulating file paths/directories
from os.path import join, exists, isdir
from os import makedirs

DATA_DIR = (
    "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/"
)

everywhere_vars = ["collection_id", 
                   "abcd_adbc01_id", 
                   "dataset_id", 
                   "subjectkey", 
                   "src_subject_id", 
                   "interview_date", 
                   "interview_age", 
                   "sex", 
                   "eventname", 
                   "visit", 
                   "imgincl_t1w_include", 
                   "imgincl_t2w_include", 
                   "imgincl_dmri_include", 
                   "imgincl_rsfmri_include", 
                   "imgincl_mid_include", 
                   "imgincl_nback_include", 
                   "imgincl_sst_include"]

changes = ['abcd_smrip10201', 'abcd_smrip20201', 'abcd_smrip30201', 
           'abcd_mrisdp10201', 'abcd_mrisdp20201', 'abcd_dti_p101', 
           'abcd_drsip101', 'abcd_drsip201', 'abcd_mrirsfd01', 
           'abcd_mrirstv02', 'abcd_betnet02', 'mrirscor02', 'abcd_tbss01']

OUT_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2/data"

# if the folder you want to save your dataset in doesn't exist, this will create it for you
if not isdir(OUT_DIR):
    makedirs(OUT_DIR)

# *exactly* as they appear in the ABCD data dictionary

variables = {
    "pdem02": [
        "demo_comb_income_v2",
        "demo_prnt_marital_v2",
        "demo_prnt_ed_v2",
    ],
    "abcd_yrb01": [
        "physical_activity1_y"
    ],
    "abcd_ssmty01": [
        "stq_y_ss_weekday", 
        "stq_y_ss_weekend"
    ],
    "acspsw03": [
        "race_ethnicity", 
        "rel_family_id",
        "sex",
        "interview_age"
    ],
    "abcd_ehis01": [
        "ehi_y_ss_scoreb"
    ],
    "abcd_lt01": [
        "site_id_l",
        "interview_date"
    ],
    "abcd_betnet02": [
        "rsfmri_c_ngd_ntpoints",
        "rsfmri_c_ngd_ad_ngd_ad",
        "rsfmri_c_ngd_ad_ngd_cgc",
        "rsfmri_c_ngd_ad_ngd_ca",
        "rsfmri_c_ngd_ad_ngd_dt",
        "rsfmri_c_ngd_ad_ngd_dla",
        "rsfmri_c_ngd_ad_ngd_fo",
        "rsfmri_c_ngd_ad_ngd_n",
        "rsfmri_c_ngd_ad_ngd_rspltp",
        "rsfmri_c_ngd_ad_ngd_smh",
        "rsfmri_c_ngd_ad_ngd_smm",
        "rsfmri_c_ngd_ad_ngd_sa",
        "rsfmri_c_ngd_ad_ngd_vta",
        "rsfmri_c_ngd_ad_ngd_vs",
        "rsfmri_c_ngd_cgc_ngd_cgc",
        "rsfmri_c_ngd_cgc_ngd_ca",
        "rsfmri_c_ngd_cgc_ngd_dt",
        "rsfmri_c_ngd_cgc_ngd_dla",
        "rsfmri_c_ngd_cgc_ngd_fo",
        "rsfmri_c_ngd_cgc_ngd_n",
        "rsfmri_c_ngd_cgc_ngd_rspltp",
        "rsfmri_c_ngd_cgc_ngd_smh",
        "rsfmri_c_ngd_cgc_ngd_smm",
        "rsfmri_c_ngd_cgc_ngd_sa",
        "rsfmri_c_ngd_cgc_ngd_vta",
        "rsfmri_c_ngd_cgc_ngd_vs",
        "rsfmri_c_ngd_ca_ngd_ca",
        "rsfmri_c_ngd_ca_ngd_dt",
        "rsfmri_c_ngd_ca_ngd_dla",
        "rsfmri_c_ngd_ca_ngd_fo",
        "rsfmri_c_ngd_ca_ngd_n",
        "rsfmri_c_ngd_ca_ngd_rspltp",
        "rsfmri_c_ngd_ca_ngd_smh",
        "rsfmri_c_ngd_ca_ngd_smm",
        "rsfmri_c_ngd_ca_ngd_sa",
        "rsfmri_c_ngd_ca_ngd_vta",
        "rsfmri_c_ngd_ca_ngd_vs",
        "rsfmri_c_ngd_dt_ngd_dt",
        "rsfmri_c_ngd_dt_ngd_dla",
        "rsfmri_c_ngd_dt_ngd_fo",
        "rsfmri_c_ngd_dt_ngd_n",
        "rsfmri_c_ngd_dt_ngd_rspltp",
        "rsfmri_c_ngd_dt_ngd_smh",
        "rsfmri_c_ngd_dt_ngd_smm",
        "rsfmri_c_ngd_dt_ngd_sa",
        "rsfmri_c_ngd_dt_ngd_vta",
        "rsfmri_c_ngd_dt_ngd_vs",
        "rsfmri_c_ngd_dla_ngd_dla",
        "rsfmri_c_ngd_dla_ngd_fo",
        "rsfmri_c_ngd_dla_ngd_n",
        "rsfmri_c_ngd_dla_ngd_rspltp",
        "rsfmri_c_ngd_dla_ngd_smh",
        "rsfmri_c_ngd_dla_ngd_smm",
        "rsfmri_c_ngd_dla_ngd_sa",
        "rsfmri_c_ngd_dla_ngd_vta",
        "rsfmri_c_ngd_dla_ngd_vs",
        "rsfmri_c_ngd_fo_ngd_fo",
        "rsfmri_c_ngd_fo_ngd_n",
        "rsfmri_c_ngd_fo_ngd_rspltp",
        "rsfmri_c_ngd_fo_ngd_smh",
        "rsfmri_c_ngd_fo_ngd_smm",
        "rsfmri_c_ngd_fo_ngd_sa",
        "rsfmri_c_ngd_fo_ngd_vta",
        "rsfmri_c_ngd_fo_ngd_vs",
        "rsfmri_c_ngd_n_ngd_n",
        "rsfmri_c_ngd_n_ngd_rspltp",
        "rsfmri_c_ngd_n_ngd_smh",
        "rsfmri_c_ngd_n_ngd_smm",
        "rsfmri_c_ngd_n_ngd_sa",
        "rsfmri_c_ngd_n_ngd_vta",
        "rsfmri_c_ngd_n_ngd_vs",
        "rsfmri_c_ngd_rspltp_ngd_rspltp",
        "rsfmri_c_ngd_rspltp_ngd_smh",
        "rsfmri_c_ngd_rspltp_ngd_smm",
        "rsfmri_c_ngd_rspltp_ngd_sa",
        "rsfmri_c_ngd_rspltp_ngd_vta",
        "rsfmri_c_ngd_rspltp_ngd_vs",
        "rsfmri_c_ngd_smh_ngd_smh",
        "rsfmri_c_ngd_smh_ngd_smm",
        "rsfmri_c_ngd_smh_ngd_sa",
        "rsfmri_c_ngd_smh_ngd_vta",
        "rsfmri_c_ngd_smh_ngd_vs",
        "rsfmri_c_ngd_smm_ngd_smm",
        "rsfmri_c_ngd_smm_ngd_sa",
        "rsfmri_c_ngd_smm_ngd_vta",
        "rsfmri_c_ngd_smm_ngd_vs",
        "rsfmri_c_ngd_sa_ngd_sa",
        "rsfmri_c_ngd_sa_ngd_vta",
        "rsfmri_c_ngd_sa_ngd_vs",
        "rsfmri_c_ngd_vta_ngd_vta",
        "rsfmri_c_ngd_vta_ngd_vs",
        "rsfmri_c_ngd_vs_ngd_vs",
    ],
    "abcd_mri01": [
        "mri_info_manufacturer",
    ],
    "abcd_imgincl01": [
        "imgincl_t1w_include", 
        "imgincl_rsfmri_include",
    ],
    "abcd_mrfindings02": [
        "mrif_score"
    ],
    "abcd_rhds01": [
        "reshist_addr1_valid",
        "reshist_addr1_proxrd",
        "reshist_addr1_popdensity",
        "reshist_addr1_urban_area",
        "reshist_addr1_pm25",
        
    ],
    "abcd_sscep01": [
        "nsc_p_ss_mean_3_items"
    ]
}

timepoints = ["baseline_year_1_arm_1"]
change_scores = False

# reads in the data dictionary mapping variables to data structures
DATA_DICT = pd.read_csv(join(DATA_DIR, 'generate_dataset/data_element_names.csv'), index_col=0)

# read in csvs of interest one a time so you don't crash your computer
# grab the vars you want, then clear the rest and read in the next
# make one "missing" column for each modality if, like RSI, a subj is missing
# on all vals if missing on one. double check this.
# also include qa column per modality and make missingness chart before/after data censoring


# IF YOU WANT LONG FORMAT DATA, LONG=TRUE, IF YOU WANT WIDE FORMAT DATA, LONG=FALSE
long = True

# initialize the progress bars
manager = enlighten.get_manager()
tocks = manager.counter(total=len(variables.keys()), desc='Data Structures', unit='data structures')

# keep track of variables that don't make it into the big df
missing = {}

# build the mega_df now
df = pd.DataFrame()
data_dict = pd.DataFrame(columns=['data_structure', 'variable_description'])
for structure in variables.keys():
        
    missing[structure] = []
    old_columns = len(df.columns)
    path = join(DATA_DIR, 'csv', f'{structure}.csv')
    if exists(path):

        # original ABCD data structures are in long form, with eventname as a column
        # but I want the data in wide form, only one row per participant
        # and separate columns for values collected at different timepoints/events
        index = ["subjectkey", "eventname"]
        cols = variables[structure]
        
        if long == True:  
            if len(timepoints) > 1:
                temp_df = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp1 = pd.DataFrame()
                for timepoint in timepoints:
                    print(timepoint)
                    temp2 = temp_df.xs(timepoint, level=1, drop_level=False)
                    temp1 = pd.concat([temp1, temp2], axis=0)
                temp_df = temp1
            else:
                temp_df0 = pd.read_csv(path, 
                              index_col="subjectkey", 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp_df = temp_df0[temp_df0['eventname'] == timepoints[0]]
                if structure in ["abcd_mrfindings02", "abcd_imgincl01", "abcd_betnet02", "abcd_lt01"]:
                    for variable in variables[structure]:
                        if variable in ["interview_date", "imgincl_t1w_include", "rsfmri_c_ngd_ntpoints", "mrif_score", "imgincl_rsfmri_include"]:
                            temp_col = temp_df0[temp_df0['eventname'] == "2_year_follow_up_y_arm_1"][variable]
                            temp_col.name = f'{variable}2'
                            temp_df = pd.concat([temp_df, temp_col], axis=1)
                        else:
                            pass
                
            df = pd.concat([df, temp_df], axis=1)
            for variable in variables[structure]:
                try:
                    data_dict.at[variable, 'data_structure'] = structure
                    data_dict.at[variable, 'variable_description'] = DATA_DICT.loc[variable, 'description']
                except Exception as e:
                    print(e)
            
        else:
            #temp_df = pd.read_csv(path, index_col="subjectkey", header=0, skiprows=[1])
            if len(timepoints) > 1:
                temp_df = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp1 = pd.DataFrame()
                for timepoint in timepoints:
                    temp2 = temp_df.xs(timepoint, level=1, drop_level=False)
                    temp_cols = [f'{col}.{timepoint}' for col in temp2.columns]
                    temp2.columns = temp_cols
                    temp1 = pd.concat([temp1, temp2], axis=1)
                temp_df = temp1
            else:
                temp_df = pd.read_csv(path, 
                              index_col="subjectkey", 
                              header=0, 
                              skiprows=[1], 
                              usecols= index + cols)
                temp_df = temp_df[temp_df['eventname'] == timepoints[0]]
            df = pd.concat([df, temp_df], axis=1)
        if change_scores:
            if structure in changes:
                path = join(DATA_DIR, 'change_scores', f'{structure}_changescores_bl_tp2.csv')
                change_cols = [f'{col}.change_score' for col in cols]
                index = ["subjectkey"]
                temp_df2 = pd.read_csv(path, 
                              index_col=index, 
                              header=0, 
                              usecols= index + change_cols)
                
                df = pd.concat([df, temp_df2], axis=1)
                
    new_columns = len(df.columns) - old_columns
    print(f"\t{new_columns} variables added!")
                
    if len(missing[structure]) >= 1:
        print(f"The following {len(missing[structure])}variables could not be added:\n{missing[structure]}")
    else:
        print(f"All variables were successfully added from {structure}.")
    temp_df = None
    tocks.update()


# how big is this thing?
print(f"Full dataframe is {sys.getsizeof(df) / 1000000}MB.")

df = df.dropna(how="all", axis=0)
df = df[df['site_id_l'] != 'site22']
df = df.loc[:,~df.columns.duplicated()].copy()

# let's grab all of the rsFC change scores
path = join(DATA_DIR, 'change_scores', 'abcd_betnet02_changescores_bl_tp2.csv')
change_scores = pd.read_csv(path, index_col="subjectkey", header=0)

# need a column for sign at baseline
for var in variables['abcd_betnet02']:
    base_change_scores = change_scores[f'{var}.baseline_year_1_arm_1'].copy()
    y2fu_change_scores = change_scores[f'{var}.2_year_follow_up_y_arm_1'].copy()
    change_scores[f'{var}.base_sign'] = np.sign(base_change_scores).copy()
    change_scores[f'{var}.2yfu_sign'] = np.sign(y2fu_change_scores).copy()
    abs_change = np.abs(y2fu_change_scores)  - np.abs(base_change_scores).copy()
    change_scores[f'{var}.change_sign'] = np.sign(abs_change).copy()

df = pd.concat([df, change_scores.filter(regex="rsfmri_c_.*", axis=1)], axis=1)

pm_factors = pd.read_excel('/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx', 
              skiprows=0, index_col=0, header=1)

noise = pd.read_csv('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/preliminary_5.0/NPS_sound/Noise_12132021.csv', 
            index_col=0, header=0, usecols=["id_redcap", "reshist_addr1_Lnight_exi"])

data_dict.at['reshist_addr1_Lnight_exi', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/preliminary_5.0/NPS_sound/Noise_12132021.csv'
data_dict.at['reshist_addr1_Lnight_exi', 
             'variable_description'] = "average total sound level from the hours of 10p-7a at primary address"

data_dict.at['F1', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F1', 
             'variable_description'] = "Factor 1 from PMF of PM2.5, crustal materials - V, Si, Ca load highest"

data_dict.at['F2', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F2', 
             'variable_description'] = "Factor 2 from PMF of PM2.5, ammonium sulfates - SO4, NH4, V load highest"

data_dict.at['F3', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F3', 
             'variable_description'] = "Factor 3 from PMF of PM2.5, biomass burning - Br, K, OC load highest"

data_dict.at['F4', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F4', 
             'variable_description'] = "Factor 4 from PMF of PM2.5, traffic (TRAP) - Fe, Cu, EC load highest"

data_dict.at['F5', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F5', 
             'variable_description'] = "Factor 5 from PMF of PM2.5, ammonium nitrates - NH4, NO3, SO4 load highest"

data_dict.at['F6', 
             'data_structure'] = '/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx'
data_dict.at['F6', 
             'variable_description'] = "Factor 6 from PMF of PM2.5, industrial fuel - Pb, Zn, Ni, Cu load highest"

data_dict.to_csv(join(OUT_DIR, 'data_dictionary.csv'))
big_df = pd.concat([df, pm_factors, noise], axis=1)
big_df.to_csv(join(OUT_DIR, 'data.csv')
)