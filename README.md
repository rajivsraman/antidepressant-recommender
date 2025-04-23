# AIPI 540 (Deep Learning Applications) - Final Project
### Author: Rajiv Raman
### Institution: Duke University
### Date: April 22, 2025

## Overview

The project task was to design a **recommendation system for antidepressant medications** powered by natural language processing. Aside from this GitHub repository, a separate 10-minute video presentation (https://youtu.be/U3UxflKeCpo) and a 3-minute pitch were organized to discuss the project. All data is sourced from the Drug Reviews dataset (from druglib.com) available through the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com). This is a dataset containing multiple patients' ratings and reviews for different prescription medications. By designing an algorithm to predict the relative rating of each medication given a patient's background, we aim to deliver the user a ranked list of the top five most relevant antidepressant medications for their needs.

1. **Naive Approach** - form a ranked list of medications based on user ratings and reviews (no user input factored in)
2. **Classical Machine Learning** - vectorize textual data in reviews via TF-IDF for training, then use linear regression to predict rating from user query
3. **Deep Learning** - generate vector embeddings and make predictions via DistilBERT, a powerful transformer model with a bidirectional encoder

The pre-trained files for the naive approach and the classical machine learning approach can be located in this repository. The pre-trained deep learning model had to be saved remotely: https://console.cloud.google.com/storage/browser/adrs-distilbert.

## Setup

In order to properly run this code, it is recommended to use Python 3.11. I believe that this code would be supported by any Python version from 3.10-3.12, but Python 3.13 will not work. I highly recommend using a virtual environment to access Python 3.11 without downgrading the full Python setup on your local machine.

1. Clone the repository - ```git clone https://github.com/rajivsraman/antidepressant-recommender.git```
2. Download the necessary requirements - ```pip install -r requirements.txt```
3. If you open scripts, you can run one of three scripts (**train_naive.py**, **train_classical.py**, **train_deep.py**) to train each of the models.
4. If you trained a model, save the new pre-trained file into the models folder.
5. You can run the demo application locally - ```streamlit run app.py```

## Application

The demo application is fully deployed along with the machine learning models. It was simple to house the linear regression model in the GitHub repository, but the DistilBERT model needed to be stored in Google Cloud. The website fully downloads these model files every time the Render application is deployed, and it can be accessed here: https://antidepressant-recommender.onrender.com/.
