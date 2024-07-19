#!/Users/katherine.b/Dropbox/Projects/Environments/idconn_env/bin python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os.path import join
from datetime import datetime
from time import strftime
from scipy.stats import spearmanr
from idconn import nbs, io
from bct import threshold_proportional


from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold, cross_validate, LeaveOneGroupOut
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap


import warnings

warnings.simplefilter("ignore")

today = datetime.today()
today_str = strftime("%m_%d_%Y")

sns.set(context="paper")

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
CONFOUNDS = ["demo_sex_v2_bl",
              "interview_age",
              "ehi_y_ss_scoreb",
              "site_id_l",
              "mri_info_manufacturer",
              "physical_activity1_y",
              "stq_y_ss_weekday",
              "stq_y_ss_weekend",
              "reshist_addr1_proxrd",
              "reshist_addr1_popdensity",
              "reshist_addr1_urban_area",
              "nsc_p_ss_mean_3_items",
              'race_ethnicity_c_bl',
              'household_income_4bins_bl',
              "rsfmri_meanmotion"
              ]
GROUPS = 'site_id_l'
TASK = "rest"
ATLAS = "gordon"
THRESH = 0.5
alpha = 0.05
K = 5
iter = 1000
atlas_fname = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/deltaABCD_clustering/resources/gordon_networks_222.nii"

num_node = 12

NETWORKS = [
    'ad', 
    'cgc', 
    'ca', 
    'dt', 
    'dla', 
    'fo', 
    #'n', 
    'rspltp', 
    'smh', 
    'smm', 
    'sa',
    'vta', 
    'vs'
]

as_factor = ["household_income_4bins_bl",
              "race_ethnicity_c_bl",
              "reshist_addr1_urban_area",
              "demo_sex_v2_bl",
              "mri_info_manufacturer",
              "site_id_l"]
upper_tri = np.triu_indices(num_node, k=0)
rsfmri_df = pd.DataFrame(index=NETWORKS, columns=NETWORKS)
for network1 in NETWORKS:
    for network2 in NETWORKS:
        var_ = f'rsfmri_c_ngd_{network1}_ngd_{network2}'
        rsfmri_df.at[network1, network2] = var_
rsfmri_vars = rsfmri_df.values[upper_tri]

dat = pd.read_pickle(join(PROJ_DIR, DATA_DIR, "data_qcd_base.pkl")).dropna()

groups = dat[GROUPS]
edges = dat[rsfmri_vars]

if CONFOUNDS is not None:
    confounds = dat[CONFOUNDS]
else:
    confounds = None
# print(dat['bc'])
for confound in CONFOUNDS:
    if type(dat.iloc[0][confound]) != np.float64:
        print(confound, type(dat.iloc[0][confound]))
        temp = pd.get_dummies(dat[confound], dtype=int, prefix=confound)
        #print(temp.columns)
        confounds = pd.concat([confounds.drop(confound, axis=1), temp], axis=1)
    elif confound in as_factor:
        print(confound)
        temp = pd.get_dummies(dat[confound], dtype=int, prefix=confound)
        #print(temp.columns)
        confounds = pd.concat([confounds.drop(confound, axis=1), temp], axis=1)


for OUTCOME in OUTCOMES:
    base_name = f"nbs-predict_outcome-{OUTCOME}"
    outcome = np.reshape(dat[OUTCOME].values, (len(dat[OUTCOME]), 1))

    weighted_average, cv_results = nbs.kfold_nbs(
        edges.values, outcome, confounds, alpha, 
        groups=groups, num_node=num_node, diagonal=True, 
        scale_x=True, scale_y=True,
        n_splits=K, n_iterations=iter
    )

    avg_df = pd.DataFrame(
        weighted_average,
        index=NETWORKS,
        columns=NETWORKS,
    )
    fig,ax = plt.subplots()
    sns.heatmap(avg_df, square=True, cmap='seismic', center=0, ax=ax)
    fig.savefig(
        f"{base_name}_weighted-{today_str}.png",
        dpi=400,
        bbox_inches='tight'
        )
    
    cv_results.to_csv(
        f"{base_name}_models-{today_str}.tsv", sep="\t"
    )
    avg_df.to_csv(
        f"{base_name}_weighted-{today_str}.tsv", sep="\t"
    )

    best = cv_results.sort_values(by='score', ascending=False).iloc[0]['model']

    # this uses the most predictive subnetwork as features in the model
    # might replace with thresholded weighted_average
    # or use _all_ the edges in weighted_average with KRR or ElasticNet...
    # ORRR use thresholded weighted average edges with ElasticNet...
    # - stays true to NBS-Predict
    # - increases parsimony while handling multicollinearity...
    # either way, I don't think cv_results is necessary
    train_metrics = {}
    if len(np.unique(outcome)) == 2:
        model = LogisticRegression(
            penalty="l2", 
            solver="saga", 
            C=best.C_[0]
            )
        train_metrics["alpha"] = best.C_[0]
        #train_metrics["l1_ratio"] = best.l1_ratio_
    else:
        model = Ridge(
            solver="saga",  
            alpha=best.alpha_
            )
        train_metrics["alpha"] = best.alpha_
        #train_metrics["l1_ratio"] = best.l1_ratio_
    # here is where we'd threshold the weighted average to use for elastic-net
    weighted_average = np.where(weighted_average > 0, weighted_average, 0)
    #nbs_vector = weighted_average[upper_tri]
    #p75 = np.percentile(nbs_vector, 75)
    #filter = np.where(nbs_vector >= p75, True, False)
    # print(nbs_vector.shape, filter.shape)
    thresh_average = threshold_proportional(weighted_average, THRESH)
    filter = np.where(thresh_average > 0, True, False)
    #print(filter)

    edge_names = rsfmri_df.values[filter]
    #print(edge_names)
    # need vectorized lower triangle
    #print(edge_names.shape, '\n\n', edge_names)
    edge_names1 = []
    for edge in edge_names:
        if edge in rsfmri_vars:
            edge_names1.append(edge)
    edges2 = dat[edge_names1]
    metrics = ['score', 'mse']
    n_splits = 10
    #rkfold = RepeatedKFold(n_splits=n_splits, n_repeats=10)
    logo = LeaveOneGroupOut()
    model_res = pd.DataFrame(
        index=range(0,logo.get_n_splits(edges2, outcome, groups=GROUPS)),
        columns=metrics
        )

    actual = {}
    predicted = {}
    for i, (train_index, test_index) in enumerate(logo.split(edges2, outcome)):
        
        edges_train =  edges2.iloc[train_index]
        outcome_train = outcome[train_index]
        confounds_train = confounds.iloc[train_index]

        edges_test =  edges2.iloc[test_index]
        outcome_test = outcome[test_index]
        confounds_test = confounds.iloc[test_index]
        # NEED TO RESIDUALIZE IF CONFOUNDS IS NOT NONE
        if CONFOUNDS is not None:
            #confounds_train = dat[CONFOUNDS].values
            outcome_train = np.reshape(outcome_train, (outcome_train.shape[0],))
            # regress out the confounds from each edge and the outcome variable,
            # use the residuals for the rest of the algorithm
            # print(confounds.shape, outcome.shape)
            if len(np.unique(outcome_train)) <= 2:
                edges_train = nbs.residualize(X=edges_train, confounds=confounds_train)
                edges_test = nbs.residualize(X=edges_test, confounds=confounds_test)
            elif len(np.unique(outcome_train)) > 3:
                outcome_train, edges_train = nbs.residualize(
                    X=edges_train.values, y=outcome_train, confounds=confounds_train.values
                )

                outcome_test, edges_test = nbs.residualize(
                    X=edges_test.values, y=outcome_test, confounds=confounds_test.values
                )

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        edges_train = x_scaler.fit_transform(edges_train)
        edges_test = x_scaler.transform(edges_test)
        if len(np.unique(outcome_train)) <= 2:
            pass
        else:
            outcome_train = y_scaler.fit_transform(outcome_train.reshape(-1, 1))
            outcome_test = y_scaler.transform(outcome_test.reshape(-1, 1))

        # run the model on the whole test dataset to get params

        # classification if the outcome is binary (for now)
        # could be extended to the multiclass case?
        
        #print(params)
        #model.set_params(**params)
        # train ElasticNet on full train dataset, using feature extraction from NBS-Predict
        fitted = model.fit(edges_train, outcome_train)
        model_res.at[i, 'score'] = fitted.score(edges_test, outcome_test)
        
        y_pred = fitted.predict(X=edges_test)
        #predicted[f"predicted {i}"] = y_pred
        #actual[f"actual {i}"] = outcome_test
        spearman = spearmanr(outcome_test, y_pred)
        model_res.at[i, 'mse'] = mean_squared_error(outcome_test, y_pred)
        model_res.at[i, 'corr'] = spearman.correlation
    #actual_df = pd.DataFrame.from_dict(actual)
    #predicted_df = pd.DataFrame.from_dict(predicted)
    #outcomes = pd.concat([actual_df, predicted_df], axis=1)
    #outcomes.to_csv(join(PROJ_DIR, OUTP_DIR, f"{base_name}_actual-predicted.tsv"), sep='\t')
    model_res.to_csv(f"{base_name}_model_performance-{today_str}.tsv", sep='\t')