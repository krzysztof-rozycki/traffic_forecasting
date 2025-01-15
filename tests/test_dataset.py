from unittest import TestCase
from forecasting.dataset import import_data, preprocess_data, make_train_test_split, split_into_x_y
from forecasting.dataset import train_config, features_config


class TestDataSet(TestCase):
    @classmethod
    def setUp(self):
        self.df_raw = import_data(train_config['input_data_path'])
        train_config['train_size'] = 0.8


    def test_preprocess_data(self):
        df = preprocess_data(self.df_raw)
        expected_columns_list = [
            'traffic_volume',
            'holiday',
            'temp',
            'rain_1h',
            'snow_1h',
            'clouds_all',
            'date_time'
        ]
        col_check = [col in df.columns for col in expected_columns_list]
        self.assertTrue(all(col_check))
        self.assertEqual(df.shape[1], 56)


    def test_split_X_y(self):
        df = preprocess_data(self.df_raw)
        X, y = split_into_x_y(df)

        self.assertEqual(len(X), len(y))
        self.assertEqual(X.ndim, 2, msg="X is not 2-dimensional")  # X should be 2-dimensional
        self.assertEqual(y.ndim, 1, msg="y is not 1-dimensional")  # y should be 1-dimensional
        self.assertNotIn(features_config['target'][0], X.columns, "Target variable found in X")


    def test_train_test_split(self):
        df = preprocess_data(self.df_raw)
        X, y = split_into_x_y(df)
        X_train, X_test, y_train, y_test = make_train_test_split(X, y)
        self.assertListEqual(list(X_train.columns), list(X_test.columns), msg="Columns of train and test are not the same")
        self.assertAlmostEqual(X_train.shape[0]/X.shape[0], 0.8, 2)
        self.assertEqual(len(X_train), len(y_train), "Length of X_train and y_train is not the same")
        self.assertEqual(len(X_test), len(y_test))
