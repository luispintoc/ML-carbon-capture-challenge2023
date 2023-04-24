# Carbon Capture Injection Rate Delta Prediction
 
This repository contains the code and results for a machine learning competition aimed at predicting carbon capture well injection rate deltas. This challenge is based on time series injection information and monitoring data from a carbon capture well. By correlating the change in injection rate to the behavior of other parameters in the well, this project can provide a checkpoint against carbon migration from the well or other losses during the process. The developed code can be used to validate carbon containment throughout the injection of the well.

# Competition Description
The Illinois Basin - Decatur Project focuses on demonstrating the capacity, injectivity, and containment of carbon storage in the Mount Simon Sandstone, which is the main carbon storage resource in the Illinois Basin and the Midwest Region. A large amount of data was collected during the three-year injection period that began in 2009. This challenge aims to apply machine learning methods to predict the injection rate delta using the first two years of injection data and corresponding fiber optic distributed temperature sensor (DTS) temperature profiles.

# Our Approach
To tackle the problem of predicting carbon capture well injection rate deltas, we have devised a two-step approach. First, we create a classifier to identify anomalies in the data, specifically when the absolute value of the target is greater than 2. This allows us to separate the problem into two distinct regimes, which we can later predict separately:

1. **Anomalies Regime**: Predicting anomalies with values ranging from -40 to -2 and 2 to 40.
2. **Low-Values Regime**: Predicting data points that lie within the -2 to 2 range.

By addressing these two regimes separately, we can tailor our models to better understand and predict each of their unique characteristics.

# Repository Structure
Please run the notebooks in order. Additionally, the last notebook '6 validation_and_submission.ipynb' is a standalone notebook since we've saved models and outputs needed to reproduce it.

## Folders
* **data/** This folder contains the input files and the files generated after the data cleaning and feature engineering process.
* **lazy_predictors/** A set of notebooks that implement various lazy predictors for the initial exploration of models.
* **metadata/** This directory stores files containing important information, such as the list of top features and missing values.
* **old_notebooks/** A collection of outdated or deprecated notebooks from earlier stages of the project.
* **output/** This folder houses output files, including saved models and prediction results.
  * **classifier/**
    * If '1' in the name, means that was trained on the first 60% of the data
    * If '2' in the name, means that was trained on the first 80% of the data
    * If '3' in the name, means that was trained on all 100% training dataset
  * **regressor_class1/**
    * If 'validation' in the name, means that it was trained with 85% of the data
    * If 'final' in the name, means that it was trained with 100% of the data
  * **regressor_class0/**
    * If 'validation' in the name, means that it was trained with 85% of the data
    * If 'final' in the name, means that it was trained with 100% of the data

* **papers/** A compilation of articles and papers related to the competition, providing valuable background information and insights.
* **sandbox/** Notebooks dedicated to experimenting with and exploring the data during the development process.

## Python notebooks/scripts
* **1 preprocessing.ipynb** A notebook focused on cleaning and feature engineering the training dataset, preparing it for model development.
* **1 preprocessing_test_set.ipynb** A notebook dedicated to cleaning and feature engineering the test dataset, ensuring it is ready for model evaluation.
* **2 xgb_classifier_feat_sel.ipynb** This notebook addresses *anomaly classification (or spikes detection)* by processing all generated features (around 3k) and selecting the top K features. It uses the first 60% of the dataset for training and the following 20% for validation (time-wise split), employing the XGBoost model.
* **3 xgb_classifier_hyperparam_opt.ipynb** A notebook that fine-tunes the classifier's hyperparameters, now using only the top K features. The model is trained on 80% of the data with 20% for validation (time-wise split).
* **4 xgb_anomalies_regressor.ipynb** This notebook focuses on *anomaly (classifier's positive class)* regression by processing all features and selecting the top K for regression. After feature selection, hyperparameter tuning is performed using the top K features.
* **5 xgb_low-values_regressor.ipynb** A notebook dedicated to *low-values (classifier's negative class)* regression, which processes all features and selects the top K for regression. After feature selection, hyperparameter tuning is performed using the top K features.
* **6 validation_and_submission.ipynb** A notebook that combines the outputs of the classifier and regressors, and prepares plots and the final submission file according to the competition's requirements.
* **utils.py** A Python file containing functions for data cleaning and feature engineering, which are used across different notebooks to maintain consistency and streamline the development process.

# License
This repository is licensed under the Apache License 2.0, a widely-used open-source software license that grants users the freedom to use, modify, and distribute the software for any purpose, both commercially and non-commercially, as well as to sublicense the software. When using this repository, you must include a copy of the Apache License 2.0, include a NOTICE file (if present), clearly state any changes made to the software, and avoid using the names of the original contributors or any trademarks they own to promote your versions of the software without their permission. By adhering to these conditions, you can freely build upon and enhance the work done by the original contributors.