import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from forecasting.utils import sin_cycle
from forecasting.enums import Period


def reformat_columns(df: pd.DataFrame):
    # changes format from object to date time
    df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
    return df


def make_features(df: pd.DataFrame):
    df = df.copy()
    df['hour'] = df['date_time'].dt.hour
    df['month'] = df['date_time'].dt.month
    df['day_of_week'] = (df['date_time'].dt.day_of_week+1)
    return df


class HighFrequencySelector(BaseEstimator, TransformerMixin):
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
    cyclical_feature = Pipeline(
        [
            (
                'sin_transformation',
                FunctionTransformer(sin_cycle, kw_args={'period': period}, feature_names_out='one-to-one')
            )
        ]
    )
    return cyclical_feature


def make_categorical_pipeline():
    categorical_features = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='constant', fill_value='None')),
            ('ohe', OneHotEncoder(min_frequency=0.1, handle_unknown='infrequent_if_exist'))
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
    dummies_features = Pipeline(
        [
            ('feature_selector', HighFrequencySelector(min_frequency=0.05))
        ]
    )
    return dummies_features


def make_col_transformer(categorical_features, numerical_features, dummies_features, cyclical_features):
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
