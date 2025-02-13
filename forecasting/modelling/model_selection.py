import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, TimeSeriesSplit, cross_val_score
from forecasting.modelling.train import make_training_pipeline
from forecasting.utils import models_config, random_search_config


def get_cv_strategy(test_size_days: int, cv_splits: int):
    """
    Defines cross-validation strategy for time series data according to number of splits and test size parameters.

    Parameters:
        test_size_days (int): number of days to be included in test fold
        cv_splits (int): number of folds used in cross-validation

    Returns:
        TimeSeriesSplit: Time-series cross-validator object
    """

    # multiply test_size_days by 24 hours to get the number of rows that will need to be placed in the test sample
    test_size = test_size_days * 24

    # return TimeSeriesSplit class that can be used as input to cross validation
    return TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)


def baseline_models_estimation(X: pd.DataFrame, y: pd.Series, cv_splits=3, scoring='neg_mean_squared_error'):
    """
    Estimates baseline models (based on models.yaml config) with default parameters. This is done to get a rough idea
    of which algorithm is more promising in estimating traffic volume problem. Later, top n performing models
    will be used in hyperparameters optimization to get the final model.

    Parameters:
        X (pd.DataFrame): A Pandas DataFrame containing (raw) features data.
        y (pd.Series): A Pandas Series containing target variable.
        cv_splits (int): An integer indication number of folds used in cross validation
        scoring (string): scoring function used to rank models, defaults to "neg_mean_squared_error"

    Returns:
        Dict: model_name: avg cv score, sorted by descending order.
    """
    cv_strategy = get_cv_strategy(random_search_config['cv_test_size_days'], cv_splits)

    results = {}
    for model_name in models_config:
        print(f"Running pipeline for {model_name}")
        model_kwargs = models_config[model_name]['default_params']
        training_pipeline = make_training_pipeline(model_name, **model_kwargs)
        cv_score = cross_val_score(training_pipeline, X, y, cv=cv_strategy, scoring=scoring)
        results[model_name] = cv_score.mean()

    results_sorted = {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}
    return results_sorted


def make_random_search(models: list, X: pd.DataFrame, y: pd.Series, scoring='neg_mean_squared_error'):
    """
    Runs random search for provided models return best value of scoring function for each model.

    Parameters:
        models (list): A list of models names. Must be one of models.yaml, otherwise it is not implemented.
        X (pd.DataFrame): A Pandas DataFrame containing (raw) features data.
        y (pd.Series): A Pandas Series containing target variable.
        scoring (string): scoring function used to rank models, defaults to "neg_mean_squared_error"

    Returns:
        Dict: model_name: RandomizedSearchCV (fitted).
    """
    results = {}
    for model_name in models:
        print(f"Running random search for {model_name}")
        training_pipeline = make_training_pipeline(model_name)
        search_space = models_config.get(model_name)['random_search']
        cv_strategy = get_cv_strategy(random_search_config['cv_test_size_days'], cv_splits=random_search_config['cv'])

        if search_space:  # doesn't run the random search if search space is not provided (for example in case of LinReg)
            # adjusting search space keys to reflect the structure of the training pipeline
            name_of_model_step = training_pipeline.steps[-1][0]
            search_space_adjusted = {name_of_model_step + "__" + key: value for key, value in search_space.items()}

            # define Random Search
            random_search = RandomizedSearchCV(
                estimator=training_pipeline,
                param_distributions=search_space_adjusted,
                n_iter=random_search_config['n_iter'],
                cv=cv_strategy,
                scoring=scoring,
                verbose=random_search_config['verbose'],
                random_state=random_search_config['random_state']
            )

            random_search.fit(X, y)

            results[model_name] = random_search

    return results
