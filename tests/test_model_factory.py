from unittest import TestCase
from sklearn.base import BaseEstimator
from forecasting.modelling.model_factory import get_model


class TestModelFactory(TestCase):
    def test_returned_type(self):
        list_of_available_models = [
        'LinearRegression',
        'Ridge',
        'ElasticNet',
        'RandomForestRegressor'
        ]

        for model_name in list_of_available_models:
            model = get_model(model_name)
            self.assertIsInstance(model, BaseEstimator)

    def test_value_error(self):
        model_name = 'not_existing_model'
        self.assertRaises(ValueError, get_model, model_name=model_name)