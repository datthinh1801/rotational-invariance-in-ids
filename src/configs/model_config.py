BATCH_SIZE = 512
EPOCHS = 30
LR = 0.001
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
            "learning_rate": LR,
        },
        "train_config": {"verbose": False},
    },
    "lgbm": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "dt": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "rf": {"hyperparams": {"max_depth": MAX_DEPTH}},
    "resnet": {"train_config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR}},
    "saint": {"train_config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE}},
    "mlp": {
        "hyperparams": {"hidden_size": 256},
        "train_config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR},
    },
}
