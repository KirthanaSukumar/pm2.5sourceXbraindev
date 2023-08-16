import json

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from os.path import join

def assign_region_names(df, missing=False):
    '''
    Input: 
    df = dataframe (variable x columns) with column containing region names in ABCD var ontology, 
    Output: 
    df = same dataframe, but with column mapping region variables to actual region names
    missing = optional, list of ABCD region names not present in region_names dictionary
    '''
    
    region_names = pd.read_csv('/Users/katherine.b/Dropbox/Projects/deltaABCD_clustering/region_names.csv', header=0, index_col=0)
    #print(region_names.index)
    # read in region names 
    missing = []
    df = df.copy()
    if not 'long_region' in df.columns:
        df['measure'] = ''
        df['region'] = ''
        df['modality'] = ''
        df['atlas'] = ''
        df['long_region'] = ''
        df['hemisphere'] = ''
        df['cog'] = ''
        df['cog2'] = ''
        df['sys'] = ''
        for var in df.index:
            #print(var)
            trim_var = var.split('.')[0]
            
            var_list = trim_var.split('_')
            
            df.at[var, 'modality'] = var_list[0]
            df.at[var, 'measure'] = var_list[1]
            df.at[var, 'atlas'] = var_list[2]
            region = '_'.join(var_list[3:])
            df.at[var, 'region'] = region
            if 'scs' in trim_var:
                if 'rsirni' in var:
                    df.at[var, 'measure'] = 'rsirnigm'
                elif 'rsirnd' in var:
                    df.at[var, 'measure'] = 'rsirndgm'
                elif '_scs_' in region:
                    temp = region.split('_scs_')
                    one = region_names.loc[temp[0]]
                    #print(one, two)
                    two = region_names.loc[temp[1]]
                    #print(one, two)
                    region_name = f'{one["name"]} {two["name"]}'
                    #print(region_name)
                    hemisphere = two['hemi']
                    df.at[var, 'long_region'] = region_name
                    df.at[var, 'hemisphere'] = hemisphere
                    df.at[var, 'measure'] = 'subcortical-network fc'
                    df.at[var, 'cog'] = f'{one["cog"]} + {two["cog"]}'
                    df.at[var, 'cog2'] = f'{one["cog2"]} + {two["cog2"]}'
                    df.at[var, 'sys'] = f'{one["sys"]} + {two["sys"]}'
                else:
                    pass
            elif '_ngd_' in region:
                temp = region.split('_ngd_')
                if temp[0] == temp[1]:
                    df.at[var, 'measure'] = 'within-network fc'
                else:
                    df.at[var, 'measure'] = 'between-network fc'
                one = region_names.loc[temp[0]]
                two = region_names.loc[temp[1]]
                region_name = f"{one['name']}-{two['name']}"
                #print(one['name'], two['name'], region_name)
                hemisphere = two['hemi']
                df.at[var, 'long_region'] = region_name
                df.at[var, 'hemisphere'] = hemisphere
                df.at[var, 'cog'] = f'{one["cog"]} + {two["cog"]}'
                df.at[var, 'cog2'] = f'{one["cog2"]} + {two["cog2"]}'
                df.at[var, 'sys'] = f'{one["sys"]} + {two["sys"]}'
            elif str(region) not in (region_names.index):
                missing.append(region)
            else:
                one = region_names.loc[region]
                df.at[var, 'long_region'] = one['name']
                df.at[var, 'hemisphere'] = one['hemi']
                df.at[var, 'cog'] = one["cog"]
                df.at[var, 'cog2'] = one["cog2"]
                df.at[var, 'sys'] = one["sys"]

        df = df[df['measure'] != 't1w']
        df = df[df['measure'] != 't2w']
    else:
        pass

    print(f'missed {len(missing)} regions bc they weren\'t in the dict')
    return df


PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/SCEHSC_Pilot/aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

sns.set(style='white', context='talk', palette='husl', font_scale=0.85)


rnd_brain_loadings = pd.read_csv(join(PROJ_DIR, OUTP_DIR, "rnd-no_site-brain_P.csv"), header=0, index_col=0)

rnd_brain_loadings = assign_region_names(rnd_brain_loadings)


rnd_brain_loadings.replace({'Thalamus': 'Temporal Pole', 
                            'limbic': 'language'}, inplace=True)


rnd_brain_loadings.replace({'motor': 'sensorimotor',
                            'reward': 'salience',
                            'interoception': 'executive function'
                           }, inplace=True)


rnd_brain_loadings.at['dmri_rsirndgm_cdk_tprh.change_score', 'cog'] = 'language'
rnd_brain_loadings.at['dmri_rsirndgm_cdk_tplh.change_score', 'cog'] = 'language'


rnd_brain_loadings[rnd_brain_loadings['long_region'] == 'Temporal Pole']


# Make the PairGrid

dat = rnd_brain_loadings.sort_values(["sys", "Dimension 1"], ascending=True)
g = sns.PairGrid(dat,
                 x_vars=dat.columns[6], y_vars=['region'],hue='sys',
                 height=15, aspect=.25, palette='husl')

# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=10, orient="h", jitter=False,
      palette="flare_r", linewidth=1, edgecolor="w")

# Use the same x axis limits on all columns and add better labels
g.set(
    xlim=(-5, 3), 
    xticks=[-5, -2.5, 0, 2.5],
    xlabel="", 
    ylabel="", 
    yticklabels=dat['long_region']
)

# Use semantically meaningful titles for the columns
titles = ["Latent dimension 1"]

for ax, title in zip(g.axes.flat, titles):

    # Set a different title for each axes
    ax.set(title=title)

    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
ax.legend(bbox_to_anchor=(-0.9, 0.73))
ax.axvline(-2.5, lw=4, ls='dotted', color="#333333", alpha=0.5)
ax.axvline(2.5, lw=4, ls='dotted', color="#333333", alpha=0.5)
sns.despine(left=True, bottom=True)

g.savefig(join(PROJ_DIR, FIGS_DIR, 'rnd-no_site-brain_region-LD1.png'), bbox_inches='tight', dpi=600)


sig_regions = rnd_brain_loadings[rnd_brain_loadings['Dimension 1'] < -2.5][['long_region', 'hemisphere', 'V1']]
sig_regions.to_csv(join(PROJ_DIR, OUTP_DIR, 'rnd-no_site-ld1-sig_brain_salience.csv'))


ap_loadings = pd.read_csv(
    join(
        PROJ_DIR, 
        OUTP_DIR, 
        "rnd-no_site-ap_source_Q.csv"), 
    header=0, 
    index_col=0
)


ap_loadings['Source'] = ap_loadings.index


ap_loadings.replace(
    {
        'NH4 SO4': 'Coal-Burning\nPower Plants',
        'NH4 NO3': 'Secondary\nNitrates'
    },
    inplace=True
)


sources = sns.crayon_palette(['Sky Blue', 'Brown', 'Red', 'Orange', 'Pink Sherbert', 'Sea Green'])


list(sources.as_hex())


fig,ax = plt.subplots(figsize=(8,6))
sns.barplot(ap_loadings, x='Source', y='Dimension 1', palette=sources)
sns.despine(bottom=True)
ax.axhline(0, lw=2, color="black", alpha=0.75)
ax.axhline(-2.5, lw=4, ls='dotted', color="#333333", alpha=0.5)
fig.savefig(join(PROJ_DIR, FIGS_DIR, 'rnd-no_site-ap_souce-LD1.png'), bbox_inches='tight', dpi=600)