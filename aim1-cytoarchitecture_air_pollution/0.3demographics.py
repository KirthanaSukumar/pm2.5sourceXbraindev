
import pandas as pd
import numpy as np

from os.path import join, isdir
from os import makedirs


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

df = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data.pkl"))
all_subj = df.index


for i in df.index:
    df.at[i, 'dmri_rsi_meanmeanmotion'] = (df.loc[i]["dmri_rsi_meanmotion"] + df.loc[i]["dmri_rsi_meanmotion2"]) / 2


before_covid = df[df['interview_date2'] < '2020-3-1'].index


valid_address = df[df['reshist_addr1_valid'] == 1].index


complete_data = pd.read_csv(join(PROJ_DIR, OUTP_DIR, 'rni-complete_sample.csv'), index_col=0)


complete_cases = complete_data.index


continuous_vars = [
    "dmri_rsi_meanmeanmotion",
    "physical_activity1_y",
    "stq_y_ss_weekday", 
    "stq_y_ss_weekend",
    "ehi_y_ss_scoreb",
    "reshist_addr1_proxrd",
    "reshist_addr1_popdensity",
    "nsc_p_ss_mean_3_items",
    "reshist_addr1_pm25",
    "reshist_addr1_Lnight_exi"
]


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


good_dmri = qc_fails[qc_fails == True].index


levels = {
    'all': all_subj, 
    'pre-covid': before_covid, 
    'valid address': valid_address, 
    'good dmri': good_dmri, 
    'complete cases': complete_cases
}


index = pd.MultiIndex.from_product([continuous_vars, ['mean', 'sdev']])

dem_df = pd.DataFrame(
    index=index,
    columns=levels.keys(),
)


for ppts in levels.keys():
    temp = df.loc[levels[ppts]]
    for var_ in continuous_vars:
        dem_df.at[(var_, 'mean'), ppts] = temp[var_].mean()
        dem_df.at[(var_, 'sdev'), ppts] = temp[var_].std()


dem_df[['all', 'complete cases']]


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
                            'Education_uptoHSGED',
                            'Education_SomeColAA',
                            'Education_Bachelors',
                            'Education_Graduate',
                            'Education_Refused',
                            'Education_Missing',
                            'MRI_Siemens', 
                            'MRI_GE', 
                            'MRI_Philips',
                            'MRI_Missing',], 
                     columns=list(levels.keys()))

for subset in levels.keys():
    #print(subset, type(col_to_df[subset]))
    temp_df = df.loc[levels[subset]]
    table.at['N', subset] = len(temp_df.index)
    table.at['Age_mean_base', subset] = np.mean(temp_df['interview_age'])
    table.at['Age_sdev_base', subset] = np.std(temp_df['interview_age'])


    # demographics
    table.at['Age_Missing', subset] = temp_df['interview_age'].isnull().sum()
    table.at['Sex_M', subset] = len(temp_df[temp_df['sex'] == 'M'].index)
    table.at['Sex_F', subset] = len(temp_df[temp_df['sex'] == 'F'].index)
    table.at['Sex_Missing', subset] = temp_df['sex'].isnull().sum()
    
    table.at['U_UrbanizedArea',
             subset] = len(temp_df[temp_df['reshist_addr1_urban_area'] == 1.].index)
    table.at['U_UrbanCluster',
             subset] = len(temp_df[temp_df['reshist_addr1_urban_area'] == 2.].index)
    table.at['U_RuralArea',
             subset] = len(temp_df[temp_df['reshist_addr1_urban_area'] == 3.].index)
    
    table.at['EHI_Right',
             subset] = len(temp_df[temp_df['ehi_y_ss_scoreb'] == 1.].index)
    table.at['EH_Mixed',
             subset] = len(temp_df[temp_df['ehi_y_ss_scoreb'] == 2.].index)
    table.at['EH_Left',
             subset] = len(temp_df[temp_df['ehi_y_ss_scoreb'] == 3.].index)
    
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



table.to_csv(join(PROJ_DIR, OUTP_DIR, 'categorical_demographcis.csv'))
