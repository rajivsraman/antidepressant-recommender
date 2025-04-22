# AIPI 540 (Deep Learning Applications) - Final Project
### Author: Rajiv Raman
### Institution: Duke University
### Date: April 22nd, 2025

## Overview

The project task was to design a **recommendation system for antidepressant medications** powered by natural language processing. Aside from this GitHub repository, a separate 10-minute video presentation and a 3-minute pitch were organized to discuss the project. All data is sourced from the Drug Reviews dataset (from druglib.com) available through the UC Irvine Machine Learning Repository (https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com). This is a dataset containing multiple patients' ratings and reviews for different prescription medications. By designing an algorithm to predict the relative rating of each medication given a patient's background, we aim to deliver the user a ranked list of the top five most relevant antidepressant medications for their needs.

1. **Naive Approach** - form a ranked list of medications based on user ratings and reviews (no user input factored in)
2. **Classical Machine Learning** - vectorize textual data in reviews via TF-IDF for training, then use linear regression to predict rating from user query
3. **Deep Learning** - generate vector embeddings and make predictions via DistilBERT, a powerful transformer model with a bidirectional encoder

## Setup

1. 
