Traffic Data 
-------------
This project focuses on building a machine learning pipeline for time series data with highly seasonal patterns. 

Dataset Description 
-------------
The project uses the I-94 Traffic Volume dataset from Kaggle, which contains hourly data on traffic volume for westbound I-94 
in Minnesota, USA. The dataset also includes weather conditions (e.g., temperature, precipitation) 
and categorical weather descriptions (e.g., Rain, Clouds, Mist), making it an ideal use case for time series 
modeling with mixed feature types.

The dataset has 48204 instances and 9 attributes. The attributes are:

 - holiday: a categorical variable that indicates whether the date is a US national holiday or a regional holiday (such as the Minnesota State Fair).
 - temp: a numeric variable that shows the average temperature in kelvin.
 - rain_1h: a numeric variable that shows the amount of rain in mm that occurred in the hour.
 - snow_1h: a numeric variable that shows the amount of snow in mm that occurred in the hour.
 - clouds_all: a numeric variable that shows the percentage of cloud cover.
 - weather_main: a categorical variable that gives a short textual description of the current weather (such as Clear, Clouds, Rain, etc.).
 - weather_description: a categorical variable that gives a longer textual description of the current weather (such as light rain, overcast clouds, etc.).
 - date_time: a datetime variable that shows the hour of the data collected in local CST time.
 - traffic_volume: a numeric variable that shows the hourly I-94 reported westbound traffic volume.

Installation
-------------
It is recommended to install the package in virtual environment. First create the environment using venv or conda and
install dependencies using ```pip install -r requirements.txt```. Then install the forecasting package by running ```pip install .```

Project Structure
------------
    ├── configs
    │   └── feature_params    <- Configs for features
    │   └── path_config          <- Configs for all needed paths
    │   └── splitting_params     <- Configs for splitting params
    │   └── train_params         <- Configs for logreg and randomforest models parametres
    │   └── predict_config.yaml  <- Config for prediction pipline
    │   └── train_config.yaml    <- Config for train pipline
    │ 
    ├── data
    │   └── raw            <- The original data dump.
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── forecasting        <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── dataset.py        <- Scripts for preprocessing and splitting dataset to train and test
    │   │
    │   ├── features.py       <- Code to create features for modeling
    │   │
    │   ├── modelling         <- Scripts to train models and then use trained models to make
    │   │   ├── model_factory.py    <- makes model based on name and parameters
    │   │   ├── model_selection.py  <- runs the process of selecting the model and tuning hyperparameters
    │   │   ├── predict.py          <- runs the prediction and evaluation of the model
    │   │   └── train.py            <- makes a training pipeline that is used for model training
    │   │   
    │   ├──  main.py        <- main script for running model selection process.
    │   │   
    │   ├──  utils.py        <- scripts for saving and loading model, importing configs etc.
    │   │
    │   ├──  enums.py        <- enum classes. currently only for number of periods for time variables
    |  
    ├── tests              <- tests for the project
    ├── setup.py           <- makes project pip installable (pip install -e .) so forecasting can be imported
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    ├── README.md          <- The top-level README for developers using this project.

Training
-----------
Run ```python main.py```
