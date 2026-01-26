<p align="center">
  <img src="logo_g4.png" width="350" title="Project Logo">
</p>

# MVP : App "G4 Detection Fraude"

This MVP  of a fraud detection learning machine model has been developed by 3 data scientist students - Selma Rochet, Sébastien Lartigue, Patrick Chardavoine - in the context of a bootcamp provided by Jedha training center.

The MVP is hosted at ["https://patrickcharda-detectionfraude.hf.space"](https://patrickcharda-detectionfraude.hf.space)

The dataset that has been choosen to build the MVP comes from Kaggle : ["https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset?select=transactions.csv"](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset?select=transactions.csv)

---

The app has 2 modes :
1. unit tests
2. bulk tests

## Unit test

Unit tests can be done by 2 ways :

    - a) automatic generation of a fraud or a legitime transaction by clicking a button that triggers a selection in an existing test dataset

    - b) get a json file that contains a fraud or a legit

## Bulk test

The user is able to pick a csv file containing a formatted dataset of online transactions.

## Assets

In the assets directory you can find :

- X_test_app.csv to try bulk test
- json files with fraud or legitime transaction
- fraud_xgb_model.pkl (xgboost model)
- a user guide (.odt)
- jupyter notebooks











