All the scripts needed to recreate the analyses reported in (Air pollution from biomass burning disrupts early adolescent cortical microarchitecture development)[https://doi.org/10.1101/2023.10.21.563430].

Run order & descriptions:
1. `0.0data_grabber.py`: Pulls the required variables from ABCD Study 4.0 data release and PMF-derived air pollution sources (Sukumaran et al., forthcoming) and organizes them in wide-form data.
2. `0.1data_cleaning.py`: Performs QC filtering on dataset.
3. `0.2missingness.py`: Assesses missingness across variables included in this project.
4. `0.3demographics.py`: Computes demographic descriptives across the whole and QC filtered datasets.
5. `1.0plsc_rn*.R`: Regresses out confounding variables and performs partial least squares correlation between RNI/RND and PM2.5 sources.
6. `1.0plsc_rn*_comps.R`: Regresses out confounding variables and performs partial least squares correlation between RNI/RND and PM2.5 components.
7. `1.2brain_plots*.py`: Creates nifti images and prototype plots with the results from #5-6.
8. `1.3loadingVis.py`: Creates PM2.5 source and brain loading plots with the results from #5-6.
9. `2.0response2reviewers.py` includes some code run to address reviewer concerns.
