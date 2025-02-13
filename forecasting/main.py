import numpy as np
from forecasting.dataset import import_data, preprocess_data, make_train_test_split, split_into_x_y
from forecasting.features import reformat_columns, make_features
from forecasting.modelling.model_selection import baseline_models_estimation, make_random_search
from forecasting.utils import save_model, train_config, features_config


def run():
    """
    Executes the entire processing pipeline.
    This function serves as the main entry point for running the end-to-end process. It handles data loading,
    preprocessing, model training, and evaluation workflows. It is primarily designed to be invoked from the CLI.

    Parameters:
        None

    Returns:
        None: This function does not return any value, but it saves the final model it the predefined directory.
    """
    # import and make initial preprocessing of the data
    print("**Processing data**\n")
    file_path = train_config['input_data_path']

    df_raw = import_data(file_path)
    df_preprocessed = preprocess_data(df_raw)
    df_reformatted = reformat_columns(df_preprocessed)
    df = make_features(df_reformatted)

    # split into X and y
    X, y = split_into_x_y(df)

    # make train test split
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    # run baseline models estimation to preselect candidates for a hyperparameters tuning
    print("**Estimating baseline models**\n")
    baseline_models = baseline_models_estimation(X_train, y_train)
    n_best_models = train_config['n_models_hyper_tuning']
    top_models = list(baseline_models.keys())[:n_best_models]

    # run random search for hyperparameters tuning for two best models
    print(f"**Running hyperparameters tunning for {n_best_models} best models**\n")
    results_rs = make_random_search(top_models, X_train, y_train)
    for model_name, estimator in results_rs.items():
        print(f"Best score for {model_name}: {estimator.best_score_}")

    # select final model and save it
    print("**Saving best model**\n")
    random_search_scores = [estimator.best_score_ for estimator in results_rs.values()]
    best_model_idx = np.argmin(np.abs(random_search_scores))

    best_model = list(results_rs.items())[best_model_idx]
    save_model(best_model, train_config['model_path'])


if __name__=="__main__":
    run()