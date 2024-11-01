# Titanic competition repository

![Cover](/references/images/Titanic_1.png)  
This repository is focused on analyzing the famous Titanic dataset 🛳️, providing insights into survival patterns 🧬 based on passenger data such as age, gender, and class. It includes Jupyter notebooks 📒 for data exploration, feature engineering, and building a machine learning classification model 🤖 to predict survival likelihood based on these variables.

## The Challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc).

## Installation
To use the code, you need to install requirements by using:
```
pip install -r requirements.txt
```
It is recommended to install requirements into a dedicated virtual environment.

## Sample results
This project is dedicated to show differences between ML classification models, and to showcase the general approach to solve classification task using Scikit-learn. For a selected model, a training curves as well as the Confusion Matrix and ROC curve are shown.
![ConfusionMatrix](/reports/figures/RandomForestsCM.png)