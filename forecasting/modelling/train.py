from sklearn.pipeline import Pipeline
from forecasting.features import make_col_transformer
from forecasting.modelling.model_factory import get_model
from forecasting.utils import features_config


def make_training_pipeline(model_name, **kwargs):
    transformer = make_col_transformer(
        categorical_features=features_config['categorical_features'],
        numerical_features=features_config['numerical_features'],
        cyclical_features=features_config['time_cyclical'],
        dummies_features=features_config['dummies_features']
    )
    model = get_model(model_name, **kwargs)

    pipeline = Pipeline(
        [
            ('transformer', transformer),
            ('model', model)
        ]
    )
    return pipeline


def train_model(model, X, y):
    model.fit(X, y)
    return model
