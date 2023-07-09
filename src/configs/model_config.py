MODEL_CONFIG = {
    "xgb": {"hyperparams": {"max_depth": 5, "verbosity": 0, "tree_method": "gpu_hist"}},
    "catboost": {
        "hyperparams": {"max_depth": 5, "task_type": "GPU", "learning_rate": 0.001},
        "train_config": {"verbose": False},
    },
    "lgbm": {"hyperparams": {"max_depth": 5}},
    "dt": {"hyperparams": {"max_depth": 5}},
    "rf": {"hyperparams": {"max_depth": 5}},
    # TODO: Change the epochs
    "resnet": {"train_config": {"epochs": 20, "batch_size": 1024, "lr": 0.001}},
    "saint": {"train_config": {"lr": 0.001, "epochs": 20, "batch_size": 1024}},
}
