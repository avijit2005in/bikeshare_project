import sys
from pathlib import Path

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikesharing_model.config.core import config
from bikesharing_model.pipeline import bikesharing_pipeline
from bikesharing_model.processing.data_manager import load_dataset, save_pipeline


def run_training() -> None:
    """
    Train the model.
    """

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(config.model_config.target, axis=1),  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )
    # Pipeline fitting

    bikesharing_pipeline.fit(X_train, y_train)
    y_pred = bikesharing_pipeline.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    # persist trained model
    save_pipeline(pipeline_to_persist=bikesharing_pipeline)
    # printing the score


if __name__ == "__main__":
    run_training()
