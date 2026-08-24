# LSST Lensed AGN Modeling
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17412879.svg)](https://doi.org/10.5281/zenodo.17412879)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.20778-b31b1b.svg)](https://arxiv.org/abs/2510.20778)


Source code of 'Lens Model Accuracy in the Expected LSST Lensed AGN Sample'. Contains network outputs, data and code to produce all figures in the paper.

In order to reproduce code from this repository, please fork it (create a copy of this repo under your GitHub account) and clone it. If you are going to run any code you should make sure that you have the requirements listed under `requirements.txt`.

1. `bash_scripts` folder: contains bash scripts that can be run on NERSC to train network or generate datasets. They work with configs stored under py_files -> `image_generation_configs` or `train_configs`.
2. `data` folder: All files must be stored under the `data` folder -- this is where the notebooks look for the data. If the data is stored in subfolders under data or elsewhere, please change the paths in the notebooks.
3. `deep-lens-modeling` folder: contains modified [lens-npe](https://github.com/smericks/lens-npe/tree/main/Inference) scripts (Erickson et al 2025).
4. `py_files` folder: contatins configuration files required for image generation (`image_generation_configs`), training (`train_configs`), NPE posterior prediction (`network_predictions.py`), final posterior  computation (`final_post.py`), utility functions (`latils.py`), image_plotting utility (`image_plotting_function.py` (credit: P. Holloway))

`catalog_creation.ipynb` produces Figures 2, 3, 4.
`npe_model_results.ipynb` produces Figure 5, 6, 8, 13 (right).
`population_level_inference.ipynb` produces Figure 7, 9, 10, 11, 12, 13 (left).

Note: Code to produce plots in `catalog_creation.ipynb` and `population_level_inference.ipynb` rely on large datasets from cosmoDC2 and MCMC chains from the hierarchical inference respectively. These files cannot be uploaded to GitHub and are hosted at this Zenodo link: [https://zenodo.org/records/17412880](https://doi.org/10.5281/zenodo.17412879)

The primary catalog that we work with (fiducal results are produced using this) is under data/fiducial_test_data.csv. The column schema for this is in test_data_metadata.csv.

Please reach out to pv10@illinois.edu if you have any questions!
