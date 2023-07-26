import logging

from pathlib import Path

import wandb
import pandas as pd
import colorlog

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from configs import *
from data_transform import preprocess_data, apply_random_rotation, feature_selection
from models import ModelFactory
from models.utils import save_model_to_file


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


def evaluate_model(
    model, X_test, y_test, class_list, iteration, rotation: bool = False
):
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

    if rotation:
        # log the results
        wandb.log(
            {
                "confusion_matrix_rotation": wandb.plot.confusion_matrix(
                    preds=y_pred,
                    y_true=y_test,
                    class_names=class_list,
                    title="Confusion matrix with rotation",
                ),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "iteration": iteration,
            }
        )
    else:
        wandb.log(
            {
                "confusion_matrix_no_rotation": wandb.plot.confusion_matrix(
                    preds=y_pred,
                    y_true=y_test,
                    class_names=class_list,
                    title="Confusion matrix without rotation",
                ),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1-score": f1,
                "iteration": iteration,
            }
        )
    return accuracy


if __name__ == "__main__":
    logger = create_logger()
    model_out_path = Path("models_out")
    if not model_out_path.exists():
        model_out_path.mkdir()

    seed = GLOBAL_CONFIG.get("seed")
    test_size = GLOBAL_CONFIG.get("test_size")
    project_name = WANDB_CONFIG.get("project")

    # loop over all datasets
    logger.info("Reading data configurations")
    data_configs = DATA_CONFIG.get("datasets", [])
    for data_config in data_configs:
        # load the data
        data_name = data_config.get("name")
        logger.info(f"Loading {data_name} dataset")
        raw_data = pd.read_csv(data_config.get("path"))

        X = raw_data.drop(columns=[data_config.get("target")])
        y = raw_data[data_config.get("target")]
        class_list = data_config.get("classes")

        # preprocess data
        logger.info("Preprocessing the data")
        X, y = preprocess_data(
            X, y, class_list, drop_cols=data_config["dropped_cols"], seed=seed
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
            # init model
            model_config = MODEL_CONFIG.get(model_name, {})
            model_hyperparams = model_config.get("hyperparams", {})
            train_config = model_config.get("train_config", {})

            # NO ROTATION
            with wandb.init(
                project=project_name,
                name=f"{data_name}_{model_name}_no_rot",
                config={"dataset": data_name, "model": model_name, "rotation": False},
            ):
                best_model = None
                best_acc = 0

                # train without rotation
                logger.info(f"Training {model_name} without rotation")
                for i in range(GLOBAL_CONFIG.get("rotation_iters")):
                    logger.info(f"Training {model_name} without rotation #{i}")
                    model = train_model(
                        model_init=init_method,
                        hyperparams=model_hyperparams,
                        train_config=train_config,
                        X_train=X_train,
                        y_train=y_train,
                    )
                    acc = evaluate_model(
                        model, X_test, y_test, class_list, iteration=i, rotation=False
                    )
                    if acc > best_acc:
                        best_acc = acc
                        best_model = model

                # save the model to file
                filepath = model_out_path / f"{model_name}_best.pkl"
                logger.info(f"Saving the model to {filepath}")
                save_model_to_file(best_model, filepath)

            # ROTATION
            logger.info("Starting rotation loop")
            with wandb.init(
                project=project_name,
                name=f"{data_name}_{model_name}_rot",
                config={
                    "dataset": data_name,
                    "model": model_name,
                    "rotation": True,
                },
            ):
                for i in range(GLOBAL_CONFIG.get("rotation_iters")):
                    logger.info(f"Rotating iteration: {i}")
                    (
                        X_train_rotated,
                        X_test_rotated,
                    ) = apply_random_rotation(X_train=X_train, X_test=X_test)

                    logger.info(f"Training {model_name} with rotation #{i}")
                    model = train_model(
                        model_init=init_method,
                        hyperparams=model_hyperparams,
                        train_config=train_config,
                        X_train=X_train_rotated,
                        y_train=y_train,
                    )
                    evaluate_model(
                        model,
                        X_test_rotated,
                        y_test,
                        class_list,
                        iteration=i,
                        rotation=True,
                    )
