# importing the tools we'll use throughout the rest of the script
# abcdWrangler is a tool for... wrangling tabulated ABCD data
import abcdWrangler as abcdw
# enlighten gives you a progress bar, but it's optional
#import enlighten
# pandas is a dataframe-managing library and it's the absolute coolest
import pandas as pd
import pyreadr
# os is more system tools, should also already be installed
# we're importing tools for verifying and manipulating file paths/directories
from os.path import join

ABCD_DIR = "/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release5.1/abcd-data-release-5.1"
PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim2"
DATA_DIR = "data"
OUTP_DIR = "output"
FIGS_DIR = "figures"

OUTCOMES = [
    "F4",
    "F5",
    "F6",
    "F3",
    "F2",
    "F1",
    ]

VARS = [
    "interview_age",
    "interview_date",
    #"site_id_l",
    "mri_info_manufacturer",
    "physical_activity1_y",
    "stq_y_ss_weekday",
    "stq_y_ss_weekend",
    "rsfmri_c_ngd_ad_ngd_ad",
    "rsfmri_c_ngd_ad_ngd_cgc",
    "rsfmri_c_ngd_ad_ngd_ca",
    "rsfmri_c_ngd_ad_ngd_dt",
    "rsfmri_c_ngd_ad_ngd_dla",
    "rsfmri_c_ngd_ad_ngd_fo",
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
    "rsfmri_c_ngd_ca_ngd_rspltp",
    "rsfmri_c_ngd_ca_ngd_smh",
    "rsfmri_c_ngd_ca_ngd_smm",
    "rsfmri_c_ngd_ca_ngd_sa",
    "rsfmri_c_ngd_ca_ngd_vta",
    "rsfmri_c_ngd_ca_ngd_vs",
    "rsfmri_c_ngd_dt_ngd_dt",
    "rsfmri_c_ngd_dt_ngd_dla",
    "rsfmri_c_ngd_dt_ngd_fo",
    "rsfmri_c_ngd_dt_ngd_rspltp",
    "rsfmri_c_ngd_dt_ngd_smh",
    "rsfmri_c_ngd_dt_ngd_smm",
    "rsfmri_c_ngd_dt_ngd_sa",
    "rsfmri_c_ngd_dt_ngd_vta",
    "rsfmri_c_ngd_dt_ngd_vs",
    "rsfmri_c_ngd_dla_ngd_dla",
    "rsfmri_c_ngd_dla_ngd_fo",
    "rsfmri_c_ngd_dla_ngd_rspltp",
    "rsfmri_c_ngd_dla_ngd_smh",
    "rsfmri_c_ngd_dla_ngd_smm",
    "rsfmri_c_ngd_dla_ngd_sa",
    "rsfmri_c_ngd_dla_ngd_vta",
    "rsfmri_c_ngd_dla_ngd_vs",
    "rsfmri_c_ngd_fo_ngd_fo",
    "rsfmri_c_ngd_fo_ngd_rspltp",
    "rsfmri_c_ngd_fo_ngd_smh",
    "rsfmri_c_ngd_fo_ngd_smm",
    "rsfmri_c_ngd_fo_ngd_sa",
    "rsfmri_c_ngd_fo_ngd_vta",
    "rsfmri_c_ngd_fo_ngd_vs",
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
    "rsfmri_meanmotion",
    "rsfmri_ntpoints",
    "imgincl_rsfmri_include",
    'mrif_score',
    "mrif_score",
    "mri_info_manufacturer",
    "mri_info_deviceserialnumber",
]
LED_VARS = [
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "reshist_addr1_urban_area",
    "nsc_p_ss_mean_3_items",
    "ehi_y_ss_scoreb",
    'reshist_addr1_Lnight_exi',
    'reshist_addr1_pm252016aa'
]

DEMO_VARS = [
    'race_ethnicity_c_bl',
    'household_income_4bins_bl',
    "site_id_l",
    'rel_family_id',
    "demo_sex_v2_bl"
]

dat = abcdw.data_grabber(ABCD_DIR, VARS, ['baseline_year_1_arm_1','2_year_follow_up_y_arm_1'], multiindex=True)
dat2 = abcdw.data_grabber(ABCD_DIR, LED_VARS, 'baseline_year_1_arm_1')
dat2.index = pd.MultiIndex.from_product([dat2.index, ['baseline_year_1_arm_1']])
demo_df = pyreadr.read_r(
    '/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/ABCD_Covariates/ABCD_release5.1/01_Demographics/ABCD_5.1_demographics_full.RDS'
)
demo_df = demo_df[None]
demo_df = demo_df[demo_df['eventname'] == 'baseline_year_1_arm_1']
demo_df.index = pd.MultiIndex.from_arrays([demo_df['src_subject_id'], demo_df['eventname']])
demo_df = demo_df.drop(['src_subject_id', 'eventname'], axis=1)
dat = pd.concat([dat, dat2, demo_df[DEMO_VARS]], axis=1)

pm_factors = pd.read_excel('/Volumes/projects_herting/LABDOCS/Personnel/Kirthana/Project3_particlePM/PMF tool results/Contribution_F6run10_fpeak.xlsx', 
              skiprows=0, index_col=0, header=1)
pm_factors.index = pd.MultiIndex.from_product([pm_factors.index, ['baseline_year_1_arm_1']])

big_df = pd.concat([dat, pm_factors], axis=1)
big_df.to_pickle(join(PROJ_DIR, DATA_DIR, 'data.pkl')
)