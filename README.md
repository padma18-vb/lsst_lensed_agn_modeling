# LSST Lensed AGN Modeling
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17412879.svg)](https://doi.org/10.5281/zenodo.17412879)

Source code of 'Lens Model Accuracy in the Expected LSST Lensed AGN Sample'. Contains network outputs, data and code to produce all figures in the paper.

In order to reproduce code from this repository, please fork it (create a copy of this repo under your GitHub account) and clone it.

All files must be stored under the data folder -- this is where the notebooks look for the data. If the data is stored in subfolders under data or elsewhere, please change the paths in the notebooks.

`catalog_creation.ipynb` produces Figures 2, 3, 4.
`npe_model_results.ipynb` produces Figure 5, 6, 8, 13 (right).
`population_level_inference.ipynb` produces Figure 7, 9, 10, 11, 12, 13 (left).

Note: Code to produce plots in `catalog_creation.ipynb` and `population_level_inference.ipynb` rely on large datasets from cosmoDC2 and MCMC chains from the hierarchical inference respectively. These files cannot be uploaded to GitHub and are hosted at this Zenodo link: [https://zenodo.org/records/17412880](https://doi.org/10.5281/zenodo.17412879)

The primary catalog that we work with (fiducal results are produced using this) is under data/fiducial_test_data.csv. The column schema for this is in test_data_metadata.csv.

Please reach out to pv10@illinois.edu if you have any questions!