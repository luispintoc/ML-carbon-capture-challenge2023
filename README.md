# Carbon Capture Injection Rate Delta Prediction
# Team name: Air Liquide

Competition link: https://xeek.ai/challenges/using-ai-to-validate-carbon-containment-in-the-illinois-basin

This repository contains the code and results for a machine learning competition aimed at predicting carbon capture well injection rate deltas. This challenge is based on time series injection information and monitoring data from a carbon capture well. By correlating the change in injection rate to the behavior of other parameters in the well, this project can provide a checkpoint against carbon migration from the well or other losses during the process. The developed code can be used to validate carbon containment throughout the injection of the well.

# Competition Description
The Illinois Basin - Decatur Project focuses on demonstrating the capacity, injectivity, and containment of carbon storage in the Mount Simon Sandstone, which is the main carbon storage resource in the Illinois Basin and the Midwest Region. A large amount of data was collected during the three-year injection period that began in 2009. This challenge aims to apply machine learning methods to predict the injection rate delta using the first two years of injection data and corresponding fiber optic distributed temperature sensor (DTS) temperature profiles.

# Our Approach
To tackle the problem of predicting carbon capture well injection rate deltas, we have devised a two-step approach. First, we create a classifier to identify anomalies in the data, specifically when the absolute value of the target is greater than 2. This allows us to separate the problem into two distinct regimes, which we can later predict separately:

1. **Anomalies Regime**: Predicting anomalies with values ranging from -40 to -2 and 2 to 40.
2. **Low-Values Regime**: Predicting data points that lie within the -2 to 2 range.

By addressing these two regimes separately, we can tailor our models to better understand and predict each of their unique characteristics.

# Repository Structure
The notebook *main.ipynb* is a standalone notebook since we've saved models and outputs needed to reproduce it.

## Folders
* **data/** This folder contains the input files and the files generated after the data cleaning and feature engineering process.
* **final_output/** This folder houses output files, including saved models and prediction results.
  * **classifier/**
  * **regressor_class1/**
  * **regressor_class0/**
* **old_notebooks/** A collection of outdated or deprecated notebooks from earlier stages of the project.
* **papers/** A compilation of articles and papers related to the competition, providing valuable background information and insights.
* **sandbox/** Notebooks dedicated to experimenting with and exploring the data during the development process.
* **temp_output/** Folder to store new output if models are retrained (to not overwrite previous work)

## Python notebooks/scripts
* **main.ipynb** Notebook with step-by-step workflow of our best solution. It has been diveded in 7 sections:
  * Section 1: Train dataset pre-processing
  * Section 2: Feature engineering
  * Section 3: Classifier
  * Section 4: Spikes regressor
  * Section 5: Low-values regressor
  * Section 6: Combining regressor's outputs in final validation set
  * Section 7: Testing
* **utils.py** A Python file containing functions for data cleaning and feature engineering, which are used across different notebooks to maintain consistency and streamline the development process.

# License
This repository is licensed under the Apache License 2.0, a widely-used open-source software license that grants users the freedom to use, modify, and distribute the software for any purpose, both commercially and non-commercially, as well as to sublicense the software. When using this repository, you must include a copy of the Apache License 2.0, include a NOTICE file (if present), clearly state any changes made to the software, and avoid using the names of the original contributors or any trademarks they own to promote your versions of the software without their permission. By adhering to these conditions, you can freely build upon and enhance the work done by the original contributors.
