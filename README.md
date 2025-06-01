# Machine Learning Cognitive Impairment Classifier

A machine learning-based web application that classifies cognitive status (CN, EMCI, LMCI, MCI) using only minimal MRI metadata — specifically age, biological gender and scan description — without requiring full image data.

# Overview
This project leverages structured metadata from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) to build a lightweight, deployable screening tool for early cognitive impairment detection. The app is built with Streamlit, making it easy for clinicians or researchers to input basic patient info and receive real-time predictions using a trained Gradient Boosting Classifier.

# Key Features
Classifies cognitive state into:

- CN (Cognitively Normal)

- EMCI (Early Mild Cognitive Impairment)

- LMCI (Late Mild Cognitive Impairment)

- MCI (General MCI)

# Minimal input requirements (age, gender, scan type)

- Trained on real clinical metadata (from ADNI-derived datasets)

- Includes preprocessing, model training, evaluation and deployment

- Web-based interface via Streamlit

- Model performance evaluated using accuracy, precision, recall, and F1-score

# Motivation
Current ML tools for Alzheimer’s rely on complex neuroimaging. This project demonstrates that even low-dimensional features can provide meaningful classification, enabling scalable triage tools for use in resource-limited settings.

# Try It Out
To run locally:

pip install -r requirements.txt

streamlit run streamlit_app.py

# Contents
- app.py: Model training and evaluation script

- streamlit_app.py: Streamlit interface

- model.pkl, model_columns.pkl: Trained model and expected features

- Data: Contains the CSV files for CN, EMCI, LMCI and MCI classes

# Dataset

This project used Alzehimer dataset from Kaggle 

Link: https://www.kaggle.com/datasets/dilipharish/alzehimercsvdatas


