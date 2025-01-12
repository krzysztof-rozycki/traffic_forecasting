import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold, TimeSeriesSplit, cross_val_score
from forecasting.modelling.train import make_training_pipeline
from forecasting.utils import models_config, random_search_config


def baseline_models_estimation(X: pd.DataFrame, y: pd.Series, cv_splits=3, scoring='neg_mean_squared_error'):
    cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    results = {}
    for model_name in models_config:
        print(f"Running pipeline for {model_name}")
        training_pipeline = make_training_pipeline(model_name)
        cv_score = cross_val_score(training_pipeline, X, y, cv=cv_strategy, scoring=scoring)
        results[model_name] = cv_score.mean()

    results_sorted = {k: v for k, v in sorted(results.items(), key=lambda item: -item[1])}
    return results_sorted


def make_random_search(models: list, X: pd.DataFrame, y: pd.Series, scoring='neg_mean_squared_error'):
    results = {}
    for model_name in models:
        print(f"Running random search for {model_name}")
        training_pipeline = make_training_pipeline(model_name)
        search_space = models_config.get(model_name)

        cv_strategy = TimeSeriesSplit(n_splits=random_search_config['cv'], test_size=365*24)

        # adjusting search space keys to reflect the structure of the training pipeline
        name_of_model_step = training_pipeline.steps[-1][0]
        if search_space:
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
