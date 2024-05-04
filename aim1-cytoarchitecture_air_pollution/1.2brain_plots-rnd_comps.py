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

def plot_surfaces(nifti, surf, cmap, vmax, threshold):
    '''
    Plots of medial and lateral left and right surface views from nifti volume
    '''
    texture_l = surface.vol_to_surf(nifti, surf.pial_left, interpolation='nearest')
    texture_r = surface.vol_to_surf(nifti, surf.pial_right, interpolation='nearest')
    
    fig = plt.figure(figsize=(12,4))
    gs = GridSpec(1, 4)

    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax1 = fig.add_subplot(gs[1], projection='3d')
    ax2 = fig.add_subplot(gs[2], projection='3d')
    ax3 = fig.add_subplot(gs[3], projection='3d')
    
    fig1 = plotting.plot_img_on_surf(
            nifti,
            cmap=cmap, 
            threshold=threshold,
            #vmax=vmax,
            symmetric_cbar=True,
            kwargs={'bg_on_data':True, 'alpha': 0.5, 'avg_method': 'max'},
            #output_file=f'../figures/{cols.name}.png'
        )

    plt.tight_layout(w_pad=-1, h_pad=-1)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         bg_map=surf.sulc_left,
                                         symmetric_cbar=True, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax0)
    figure = plotting.plot_surf_stat_map(surf.pial_left, 
                                         texture_l, 
                                         bg_map=surf.sulc_left,
                                         symmetric_cbar=True, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax1)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         bg_map=surf.sulc_right,
                                         symmetric_cbar=True, 
                                         threshold=threshold,
                                         cmap=cmap, 
                                         view='lateral', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax2)
    figure = plotting.plot_surf_stat_map(surf.pial_right, 
                                         texture_r, 
                                         bg_map=surf.sulc_right,
                                         symmetric_cbar=True, 
                                         threshold=threshold,     
                                         cmap=cmap, 
                                         view='medial', 
                                         colorbar=False, 
                                         vmax=vmax, 
                                         axes=ax3)
    return figure, fig1

def series_2_nifti(series_in, out_dir, save=False):
    nifti_mapping = pd.read_pickle('/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/data/variable_to_nifti_mapping.pkl')
    series = series_in.copy()
    series.index = [x.split('.')[0] for x in series.index]
    
    #vmin = series.quantile(0.25)

    # list of measures to plot
    measures = {'cortical-thickness': 'smri_thick_cdk_.*',
                'cortical-gwcontrast': 'smri_t1wcnt_cdk_.*',
                'cortical-area': 'smri_area_cdk_.*',
                'cortical-volume': 'smri_vol_cdk_.*', 
                'subcortical-volume': 'smri_vol_scs_.*', 
                'subcortical-RND': 'dmri_rsirnd_scs_.*',
                'subcortical-RNI': 'dmri_rsirni_scs_.*',
                'cortical-RND': 'dmri_rsirndgm_.*',
                'cortical-RNI': 'dmri_rsirnigm_.*',
                'cortical-BOLD-variance': 'rsfmri_var_cdk_.*',
                'tract-volume': 'dmri_dtivol_fiberat_.*', 
                'tract-FA': 'dmri_dtifa_fiberat_.*', 
                'tract-MD': 'dmri_dtimd_fiberat_.*',
                'tract-LD': 'dmri_dtild_fiberat_.*', 
                'tract-TD': 'dmri_dtitd_fiberat_.*', 
                'tract-RND': 'dmri_rsirnd_fib_.*',
                'tract-RNI': 'dmri_rsirni_fib_.*'}
    fc_cort_var = series.filter(regex='.*fmri.*_c_.*').index
    fc_scor_var = series.filter(regex='.*fmri.*_cor_.*').index
    fmri_var_var = series.filter(regex='.*fmri.*_var_.*').index

    #morph_var = df[df['concept'] == 'macrostructure'].index
    #cell_var = df[df['concept'] == 'microstructure'].index
    func_var = list(fmri_var_var) 
    conn_var = list(fc_cort_var) + list(fc_scor_var)

    conn_measures = {'cortical-network-connectivity': 'rsfmri_c_ngd_.*',
                'subcortical-network-connectivity': 'rsfmri_cor_ngd_.*_scs_.*',}

    # let's plot APC on brains pls
    for measure in measures.keys():
        #print(measure, measures[measure])
        #print(measure)

        meas_df = series.filter(regex=measures[measure], axis=0)
        meas_vars = meas_df.index

        #meas_df.drop_duplicates(inplace=True)
        #print(len(meas_df.index))
        #print(meas_df.head())
        if len(meas_df[meas_df != 0]) == 0:
            pass
        else:
            if 'tract' in measure:
                #print('tract')
                fibers = nifti_mapping.filter(regex=measures[measure], axis=0).index
                var = fibers[0]
                tract_fname = nifti_mapping.loc[var]['atlas_fname']
                tract_nii = nib.load(tract_fname)
                tract_arr = tract_nii.get_fdata()
                #print(np.unique(tract_arr))
                avg = series.loc[f'{var}']
                tract_arr *= avg
                all_tracts_arr = np.zeros(tract_arr.shape)
                all_tracts_arr += tract_arr
                for var in fibers[1:]:    
                    tract_fname = nifti_mapping.loc[var]['atlas_fname']
                    if type(tract_fname) is str:
                        try:
                            tract_nii = nib.load(tract_fname)
                            tract_arr = tract_nii.get_fdata()
                            #print(np.unique(tract_arr))
                            avg = series.loc[f'{var}']
                            tract_arr *= avg
                            all_tracts_arr += tract_arr
                        except Exception as e:
                            pass
                    else:
                        pass
                meas_nimg = nib.Nifti1Image(all_tracts_arr, tract_nii.affine)
                if save:
                    meas_nimg.to_filename(f'{out_dir}/{series.name}.nii')
                
            else:
                #print('cortex')
                #print(nifti_mapping.loc[meas_vars]['atlas_fname'])
                atlas_fname = nifti_mapping.loc[meas_vars]['atlas_fname'].unique()[0]
                #print(atlas_fname)
                atlas_nii = nib.load(atlas_fname)
                atlas_arr = atlas_nii.get_fdata()
                plotting_arr = np.zeros(atlas_arr.shape)
                for i in meas_df.index:
                    if i in nifti_mapping.index:
                        value = nifti_mapping.loc[i]['atlas_value']
                        
                        #print(i, value)
                        if value is np.nan:
                            pass
                        
                        else:
                            val = series.at[i]
                            #print(avg, value, atlas_arr.shape)
                            plotting_arr[np.where(atlas_arr == value)] = val
                    else:
                        pass
                
                meas_nimg = nib.Nifti1Image(plotting_arr, atlas_nii.affine)
                #print(np.mean(plotting_arr))
                if save:
                    meas_nimg.to_filename(f'{out_dir}/{series.name}.nii')

    
    return meas_nimg

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

#print(sig_dims)
cmap = 'seismic'
threshold = 2

rnd_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rnd-brain_P-components.csv"), header=0, index_col=0)

rnd_brain_loadings = assign_region_names(rnd_brain_loadings)

sig_dims = rnd_brain_loadings.filter(regex="Dimension.*").columns
#print(sig_dims)

# mask loadings with significance
for sig_dim in sig_dims:
    
    #print(dim_idx)
    loadings = rnd_brain_loadings[sig_dim]
    #print(loadings.sort_values())
    # when less than 2, mask = True
    # and true values are replaced
    # false values (>2) are retained
    #mask = abs(rnd_brain_loadings[sig_dim]) < 2.5
    #print(sum(mask))
    #masked_loadings = loadings.mask(mask)
    loadings.name = sig_dim.replace(' ', '_')
    nifti = series_2_nifti(loadings, join(PROJ_DIR, OUTP_DIR), save=True)
    plotting.plot_img_on_surf(
            nifti,
            cmap=cmap, 
            threshold=threshold,
            #vmax=vmax,
            symmetric_cbar=True,
            kwargs={'bg_on_data':False, 'alpha': 1, 'avg_method': 'max'},
            output_file=join(PROJ_DIR, FIGS_DIR, f'rnd-components_{loadings.name}.png')
        )

rnd_brain_loadings.to_csv(join(PROJ_DIR, OUTP_DIR, "rnd-brain_P-components-region_names.csv"))

