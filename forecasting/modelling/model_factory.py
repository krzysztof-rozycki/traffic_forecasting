from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor


def get_model(model_name: str, **kwargs):
    model_mapping = {
        'LinearRegression': LinearRegression,
        'Ridge': Ridge,
        'ElasticNet': ElasticNet,
        'RandomForestRegressor': RandomForestRegressor
    }
    model = model_mapping.get(model_name)

    if model is None:
        raise ValueError(f"Model {model_name} is not implemented")

    return model(**kwargs)
