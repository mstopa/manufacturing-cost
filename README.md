# Bosh production line performance

## Project setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Download Kaggle's data from [source](https://www.kaggle.com/competitions/bosch-production-line-performance) or follow `data/README.md` to use Kaggle API.

The data represents measurements of parts as they move through Bosch's production lines.\
Each part has a unique Id.\
The goal is to predict which parts will fail quality control (represented by a 'Response' = 1).\

Features are named according to a convention:
```
L3_S36_F3939
```
`L3` - line 3\
`S36` - station 36\
`F3939` - feature number 3939

| File Name       | Description                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------- |
| train_numeric.csv | the training set numeric features (this file contains the 'Response' variable)                    |
| test_numeric.csv  | the test set numeric features (you must predict the 'Response' for these Ids)                     |
| train_categorical.csv | the training set categorical features                                                          |
| test_categorical.csv  | the test set categorical features                                                              |
| train_date.csv   | the training set date features                                                                    |
| test_date.csv    | the test set date features                                                                       | 
| sample_submission.csv | a sample submission file in the correct format                                                 |

In addition to being one of the largest datasets (in terms of number of features) ever hosted on Kaggle, the ground truth for this competition is highly imbalanced. Together, these two attributes are expected to make this a challenging problem.


### Setup for notebooks
1.
```bash
python3 -m venv .venv
ipython kernel install --name <KERNEL_NAME> --user
```
2. After opening the notebook, make sure to switch to the custom kernel.