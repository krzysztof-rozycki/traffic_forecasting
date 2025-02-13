import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from forecasting.utils import cos_cycle, sin_cycle, train_config
from forecasting.enums import Period


def reformat_columns(df: pd.DataFrame):
    # changes format of date_time from object to date time
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
    return df


def make_features(df: pd.DataFrame):
    # add new features:
    #   - hour (int): hour of the day
    #   - month (int): month of the year
    #   - day_of_week (int): number indicating what is the day of the week
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['month'] = df['date_time'].dt.month
    df['day_of_week'] = (df['date_time'].dt.day_of_week+1)
    return df


class HighFrequencySelector(BaseEstimator, TransformerMixin):
    """
    Class to be used as part of the modelling pipeline. It is supposed to do dummy features selection based on minimum
    frequency.
    """
    def __init__(self, min_frequency=0.05):
        self.min_frequency = min_frequency
        self.selected_columns_ = None

    def fit(self, X, y=None):
        frequencies = (X.sum() / len(X))
        self.selected_columns_ = frequencies[frequencies >= self.min_frequency].index.tolist()
        return self

    def transform(self, X):
        return X[self.selected_columns_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_columns_


def make_cyclical_feature_pipeline(period):
    """
    Builds pipeline for cyclical features. Cyclical features are hour, month and day_of_week. They shouldn't be used
    directly with the model as, for example, in case of "hour" the values 23 and 0 are far apart from each other, while
    in fact they are next to each other in a day cycle. Therefore, these features are processed with the cos_cycle or
    sin_cycle function which ensures that the values 23 and 0 are close to each other.

    Parameters:
        period (int): A value indicating what is the length of a full cycle, for example 24 in case of the hour or
            12 in case of the month

    Returns:
        Pipeline: sklearn Pipeline with FunctionTransformer.
    """
    transformations = [
        (
            'sin_transformation',
            FunctionTransformer(sin_cycle, kw_args={'period': period}, feature_names_out='one-to-one')
        ),
        (
            'cos_transformation',
            FunctionTransformer(cos_cycle, kw_args={'period': period}, feature_names_out='one-to-one')
        )
    ]

    cyclical_feature = Pipeline(
        [
            ('cyclical_transformation', FeatureUnion(transformations))
        ]
    )
    return cyclical_feature


def make_categorical_pipeline():
    categorical_features = Pipeline(
        [
            ('imputer', SimpleImputer(
                strategy='constant',
                fill_value='None'
            )),
            ('ohe', OneHotEncoder(
                min_frequency=train_config['min_frequency_ohe'],
                handle_unknown='infrequent_if_exist'
            ))
        ]
    )
    return categorical_features


def make_numeric_pipeline():
    numeric_features = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]
    )
    return numeric_features


def make_dummies_pipeline():
    """
    Builds pipeline for dummy features. These are features made of weather_main and weather_description variables from
    the raw data. These two features are often duplicated for the same date_time record, which means that they can't
    be process by the regular OneHotEncoder class, but rather need to be preprocessed by a separate process beforehand
    to ensure one date_time value has only one record in the data.

    Parameters:
        None

    Returns:
        Pipeline: sklearn Pipeline with HighFrequencySelector.
    """
    dummies_features = Pipeline(
        [
            ('feature_selector', HighFrequencySelector(min_frequency=train_config['min_frequency_weather_dummies']))
        ]
    )
    return dummies_features


def make_col_transformer(categorical_features, numerical_features, dummies_features, cyclical_features):
    # for cyclical features create pipeline with sin and cos cycle transformation
    cyclical_transformers = []
    for col in cyclical_features:
        period = Period[col].value
        cyclical_transformers.append((col, make_cyclical_feature_pipeline(period), col))

    col_transformer = ColumnTransformer(
        [
            ('categorical', make_categorical_pipeline(), categorical_features),
            ('numeric', make_numeric_pipeline(), numerical_features),
            ('dummies', make_dummies_pipeline(), dummies_features),
            *cyclical_transformers
        ]
    )
    return col_transformer
