
# importing the tools we'll use throughout the rest of the script
# sys is system tools, should already be installed
import sys
import json
# pandas is a dataframe-managing library and it's the absolute coolest
import pandas as pd
# numpy is short for "numerical python" and it does math
import numpy as np
# seaborn is a plotting library named after a character from West Wing
# it's kind of like python's ggplot
import seaborn as sns
# nibabel handles nifti images
import nibabel as nib

# os is more system tools, should also already be installed
# we're importing tools for verifying and manipulating file paths/directories
from os.path import join, exists, isdir
from os import makedirs

# nilearn makes the best brain plots
# and their documentation/examples are so, so handy
# https://nilearn.github.io/stable/auto_examples/01_plotting/index.html
from nilearn import plotting, surface, datasets

# matplotlib is the backbone of most python plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
# gridspec helps us put lots of panels on one figure
from matplotlib.gridspec import GridSpec

def assign_region_names(df, missing=False):
    '''
    Input: 
    df = dataframe (variable x columns) with column containing region names in ABCD var ontology, 
    Output: 
    df = same dataframe, but with column mapping region variables to actual region names
    missing = optional, list of ABCD region names not present in region_names dictionary
    '''
    
    # read in region names 
    with open('/Volumes/projects_herting/LABDOCS/PROJECTS/ABCD/Data/release4.0/python_scripts/abcd_vars_to_region_names.json') as json_file:
        region_names = json.load(json_file)
    
    missing = []
    if not 'long_region' in df.columns:
        df['measure'] = ''
        df['region'] = ''
        df['modality'] = ''
        df['atlas'] = ''
        df['long_region'] = ''
        df['hemisphere'] = ''
        for var in df.index:
            #print(var)
            if 'mrisdp' in var:
                var_num = int(var.split('.')[0].split('_')[-1])
                df.at[var, 'modality'] = 'smri'
                df.at[var, 'atlas'] = 'dtx'
                if var_num <= 148:
                    df.at[var, 'measure'] = 'thick'
                elif var_num <= 450 and var_num >= 303:
                    df.at[var, 'measure'] = 'area'
                elif var_num < 604 and var_num >= 450:
                    df.at[var, 'measure'] = 'vol'
                elif var_num <= 1054 and var_num >= 907:
                    df.at[var, 'measure'] = 't1wcnt'
                elif var_num == 604:
                    df.at[var, 'measure'] = 'gmvol'
            elif '_' in var:
                var_list = var.split('.')[0].split('_')
                df.at[var, 'modality'] = var_list[0]
                df.at[var, 'measure'] = var_list[1]
                df.at[var, 'atlas'] = var_list[2]
                region = '_'.join(var_list[3:])
                df.at[var, 'region'] = region
                if 'scs' in var:
                    if 'rsirni' in var:
                        df.at[var, 'measure'] = 'rsirnigm'
                    elif 'rsirnd' in var:
                        df.at[var, 'measure'] = 'rsirndgm'
                    else:
                        pass
                else:
                    pass
                if '_scs_' in region:
                    temp = region.split('_scs_')
                    region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
                    hemisphere = region_names[temp[1]][1]
                    df.at[var, 'long_region'] = region_name
                    df.at[var, 'hemisphere'] = hemisphere
                    df.at[var, 'measure'] = 'subcortical-network fc'
                elif '_ngd_' in region:
                    temp = region.split('_ngd_')
                    if temp[0] == temp[1]:
                        df.at[var, 'measure'] = 'within-network fc'
                    else:
                        df.at[var, 'measure'] = 'between-network fc'
                    region_name = f'{region_names[temp[0]][0]}, {region_names[temp[1]][0]}'
                    hemisphere = region_names[temp[1]][1]
                    df.at[var, 'long_region'] = region_name
                    df.at[var, 'hemisphere'] = hemisphere
                elif str(region) not in (region_names.keys()):
                    missing.append(region)
                else:
                    long_region = region_names[region]
                    df.at[var, 'long_region'] = long_region[0]
                    df.at[var, 'hemisphere'] = long_region[1]

        df = df[df['measure'] != 't1w']
        df = df[df['measure'] != 't2w']
    else:
        pass
    
    print(f'missed {len(missing)} regions bc they weren\'t in the dict')
    return df

def jili_sidak_mc(data, alpha):
    '''
    Accepts a dataframe (data, samples x features) and a type-i error rate (alpha, float), 
    then adjusts for the number of effective comparisons between variables
    in the dataframe based on the eigenvalues of their pairwise correlations.
    '''
    import math
    import numpy as np

    mc_corrmat = data.corr()
    mc_corrmat.fillna(0, inplace=True)
    eigvals, eigvecs = np.linalg.eig(mc_corrmat)

    M_eff = 0
    for eigval in eigvals:
        if abs(eigval) >= 0:
            if abs(eigval) >= 1:
                M_eff += 1
            else:
                M_eff += abs(eigval) - math.floor(abs(eigval))
        else:
            M_eff += 0
    print('\nFor {0} vars, number of effective comparisons: {1}\n'.format(mc_corrmat.shape[0], M_eff))

    #and now applying M_eff to the Sidak procedure
    sidak_p = 1 - (1 - alpha)**(1/M_eff)
    if sidak_p < 0.00001:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:2e} after corrections'.format(sidak_p))
    else:
        print('Critical value of {:.3f}'.format(alpha),'becomes {:.6f} after corrections'.format(sidak_p))
    return sidak_p, M_eff

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

rnd_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rnd-brain_P-components.csv"), header=0, index_col=0)
rni_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rni-brain_P-components.csv"), header=0, index_col=0)

rnd_brain_loadings = assign_region_names(rnd_brain_loadings)

# print the number of effective comparisons (here: will be number of LVs bc they're orthogonal)
jili_sidak_mc(rnd_brain_loadings.filter(like='V'), 0.01)
jili_sidak_mc(rni_brain_loadings.filter(like='V'), 0.01)


rni2_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rni-brain_P.csv"), header=0, index_col=0)
rnd2_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rnd-brain_P.csv"), header=0, index_col=0)

jili_sidak_mc(rni2_brain_loadings.filter(like='V'), 0.01)
jili_sidak_mc(rnd2_brain_loadings.filter(like='V'), 0.01)

# to calculate the average elapsed time between visits, grab age data from each ppt at each time point
ppts = pd.read_csv(join(PROJ_DIR, OUTP_DIR, 'plsc_ppt_ids.csv'))
ppts = ppts['x']

big_dat = pd.read_pickle('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/data.pkl')

# calculate descriptive statistics for elapsed time in this analytic sample
elapsed_time = big_dat.loc[ppts.values]['interview_age.2_year_follow_up_y_arm_1'] - big_dat.loc[ppts.values]['interview_age.baseline_year_1_arm_1']
elapsed_time.describe().to_csv(join(PROJ_DIR, OUTP_DIR, 'sample_age_change.csv'))

# calculate descriptive statistics for elapsed time in the whole ABCD Study sample
big_elapsed_time = big_dat['interview_age.2_year_follow_up_y_arm_1'] - big_dat['interview_age.baseline_year_1_arm_1']
big_elapsed_time.describe().to_csv(join(PROJ_DIR, OUTP_DIR, 'abcd_age_change.csv'))

# calculate descriptives of regional changes in isotropic (RNI) and anisotropic (RND) intracellular diffusion
rni_desc = big_dat.filter(regex="dmri_rsirnigm_cdk.*change_score").loc[ppts].describe()
rnd_desc = big_dat.filter(regex="dmri_rsirndgm_cdk.*change_score").loc[ppts].describe()

rni_means = rni_desc.T['mean']
rni_means.name = 'RNI'
rnd_means = rnd_desc.T['mean']
rnd_means.name = 'RND'

pd.concat([rni_means, rnd_means], axis=1).to_csv(join(PROJ_DIR, OUTP_DIR, 'rsi_means.csv'))

rni_desc = assign_region_names(rni_desc.T)
rni_desc.index = rni_desc['long_region']
rni_desc.to_csv(join(PROJ_DIR, OUTP_DIR, 'rni_change_descriptives.csv'))

rnd_desc = assign_region_names(rnd_desc.T)
rnd_desc.index = rnd_desc['long_region']
rnd_desc.to_csv(join(PROJ_DIR, OUTP_DIR, 'rnd_change_descriptives.csv'))

# redo the distribution plots for rni and rnd
cell_cmap = sns.diverging_palette(71, 294.3, s=70, l=50, center="light", n=6, as_cmap=False)
sns.set(context='talk', style='white')

fig,ax = plt.subplots(figsize=(6,1))
g = sns.kdeplot(rni_means, fill=True, color=cell_cmap.as_hex()[-1])
h = sns.kdeplot(rnd_means, fill=True, color=cell_cmap.as_hex()[0])
ax.set_xlabel('Annual change (%)')
ax.set_ylabel('')
sns.despine()
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rsi_change_dist.svg'), bbox_inches='tight')