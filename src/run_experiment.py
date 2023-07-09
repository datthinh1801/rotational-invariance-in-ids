import logging
import pickle

from pathlib import Path
from datetime import datetime

import ipdb
import wandb
import pandas as pd
import colorlog

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from configs import *
from data_transform import preprocess_data, apply_random_rotation, feature_selection
from models import ModelFactory


def create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()

    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s%(reset)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def save_model_to_file(model, filename):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model_from_file(filename):
    with open(filename, "rb") as file:
        model = pickle.load(file)
    return model


def train_model(
    model_init,
    hyperparams: dict,
    train_config: dict,
    X_train,
    y_train,
):
    model = model_init(**hyperparams)
    model.fit(X_train, y_train, **train_config)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a model on test data and calculate accuracy, precision, recall, and f1 score.

    Args:
    - model: The trained model to evaluate.
    - X_test: The feature matrix of the test data.
    - y_test: The target vector of the test data.

    Returns:
    - accuracy: Accuracy score of the model on the test data.
    - precision: Precision score of the model on the test data.
    - recall: Recall score of the model on the test data.
    - f1_score: F1 score of the model on the test data.
    """
    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return accuracy, precision, recall, f1


if __name__ == "__main__":
    logger = create_logger()
    model_out_path = Path("models_out")
    if not model_out_path.exists():
        model_out_path.mkdir()

    seed = GLOBAL_CONFIG.get("seed")
    test_size = GLOBAL_CONFIG.get("test_size")
    wandb_run_name_suffix = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # loop over all datasets
    logger.info("Reading data configurations")
    data_configs = DATA_CONFIG.get("datasets", [])
    for data_config in data_configs:
        # load the data
        logger.info(f"Loading {data_config.get('name')} dataset")
        raw_data = pd.read_csv(data_config.get("path"))

        X = raw_data.drop(columns=[data_config.get("target")])
        y = raw_data[data_config.get("target")]

        # preprocess data
        logger.info("Preprocessing the data")
        X, y, label_encoder = preprocess_data(
            X, y, drop_cols=data_config["dropped_cols"], seed=seed
        )

        # feature selection
        # logger.info("Selecting best features")
        # X, y = feature_selection(X, y, seed=seed)

        # train test split
        logger.info("Splitting train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

        # load model
        model_factory = ModelFactory()
        for model_name, init_method in model_factory.loop_model():
            WANDB_CONFIG[
                "name"
            ] = f"{data_config.get('name')}_{model_name}_{wandb_run_name_suffix}"
            wandb.init(**WANDB_CONFIG)
            wandb_config = {
                "data_name": data_config.get("name"),
                "model": model_name,
                "n_features": X.shape[1],
                "test_size": test_size,
                "seed": seed,
            }

            # init model
            model_config = MODEL_CONFIG.get(model_name, {})
            model_hyperparams = model_config.get("hyperparams", {})
            train_config = model_config.get("train_config", {})

            wandb_config.update(model_hyperparams)
            wandb_config.update(train_config)

            # train without rotation
            logger.info(f"Training {model_name} without rotation")
            model = train_model(
                model_init=init_method,
                hyperparams=model_hyperparams,
                train_config=train_config,
                X_train=X_train,
                y_train=y_train,
            )

            # save the model to file
            filepath = model_out_path / f"{model_name}.pkl"
            logger.info(f"Saving the model to {filepath}")
            save_model_to_file(model, filepath)

            accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
            logger.info(
                f"Accuracy: {accuracy} - Precision: {precision} - Recall: {recall} - F1-score: {f1}"
            )
            wandb.log(
                {
                    "iteration": 0,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

            logger.info("Starting rotation loop")
            for iteration_idx in range(1, GLOBAL_CONFIG.get("rotation_iters") + 1):
                logger.info(f"Rotating iteration: {iteration_idx}")
                (
                    X_train_rotated,
                    X_test_rotated,
                ) = apply_random_rotation(X_train=X_train, X_test=X_test)

                logger.info(f"Training {model_name} with rotation")
                model = train_model(
                    model_init=init_method,
                    hyperparams=model_hyperparams,
                    train_config=train_config,
                    X_train=X_train_rotated,
                    y_train=y_train,
                )
                accuracy, precision, recall, f1 = evaluate_model(
                    model, X_test_rotated, y_test
                )
                logger.info(
                    f"Accuracy: {accuracy} - Precision: {precision} - Recall: {recall} - F1-score: {f1}"
                )
                wandb.log(
                    {
                        "iteration": iteration_idx,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )

            wandb.finish()
