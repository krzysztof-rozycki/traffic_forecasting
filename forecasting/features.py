import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
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


def make_col_transformer(categorical_features, numerical_features, cyclical):
    cyclical_transformers = []
    for col in cyclical:
        period = Period[col].value
        cyclical_transformers.append((col, make_cyclical_feature_pipeline(period), col))

    col_transformer = ColumnTransformer(
        [
            ('categorical', make_categorical_pipeline(), categorical_features),
            ('numeric', make_numeric_pipeline(), numerical_features),
            *cyclical_transformers
        ]
    )
    return col_transformer
