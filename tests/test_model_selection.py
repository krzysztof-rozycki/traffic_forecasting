import unittest
from forecasting.modelling.model_selection import get_cv_strategy
from sklearn.model_selection import TimeSeriesSplit


class TestCVStrategy(unittest.TestCase):
    def test_get_cv_strategy(self):
        # Define the input parameters
        test_size_days = 7
        cv_splits = 5
        cv_strategy = get_cv_strategy(test_size_days, cv_splits)

        # Assert that the returned object is indeed an instance of TimeSeriesSplit
        self.assertIsInstance(cv_strategy, TimeSeriesSplit, "The returned object should be an instance of TimeSeriesSplit.")

        # Assert the configured properties are as expected
        self.assertEqual(cv_strategy.n_splits, cv_splits, "The number of splits is incorrect.")
        self.assertEqual(cv_strategy.test_size, test_size_days * 24, "The test size (in hours) is incorrect.")
