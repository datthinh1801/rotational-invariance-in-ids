from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .resnet import ResNet
from .saint import SAINTModel
from .mlp import MLPModel


class ModelFactory:
    def __init__(self) -> None:
        self.clf_table = {
            "saint": SAINTModel,
            "resnet": ResNet,
            "mlp": MLPModel,
            "xgb": XGBClassifier,
            "catboost": CatBoostClassifier,
            "lgbm": LGBMClassifier,
            "dt": DecisionTreeClassifier,
            "rf": RandomForestClassifier,
        }

    def loop_model(self):
        for model_name in self.clf_table.keys():
            yield model_name, self.clf_table.get(model_name)
