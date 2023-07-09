EPOCHS = 1
MAX_DEPTH = 5
MODEL_CONFIG = {
    "xgb": {
        "hyperparams": {
            "max_depth": MAX_DEPTH,
            "verbosity": 0,
            "tree_method": "gpu_hist",
        }
    },
    "catboost": {
        "hyperparams": {
            "max_depth": MAX_DEPTH,
            "task_type": "GPU",
            "learning_rate": 0.001,
        },
        "train_config": {"verbose": False},
    },
    "lgbm": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "dt": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "rf": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "resnet": {"train_config": {"epochs": EPOCHS, "batch_size": 1024, "lr": 0.001}},
    "saint": {"train_config": {"lr": 0.001, "epochs": EPOCHS, "batch_size": 1024}},
    "mlp": {
        "hyperparams": {"hidden_size": 256},
        "train_config": {"epochs": EPOCHS, "batch_size": 1024},
    },
}
