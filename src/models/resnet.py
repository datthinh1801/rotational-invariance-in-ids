import ipdb

import timm
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from sklearn.metrics import accuracy_score

from .utils import save_model_to_file, load_model_from_file


class ResNet:
    def __init__(
        self,
    ):
        self.model_name = "resnet"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use: {self.device}")

        # num_classes is set to 1 to be compatible with the BCEWithLogitLoss
        # which requires num_classes=1 for single-label binary classification
        self.model = timm.create_model("resnet18", pretrained=False, num_classes=1).to(
            self.device
        )

    def _process_input(self, X):
        n_samples, n_features = X.shape
        X = torch.from_numpy(X).to(self.device).to(torch.float32)
        X = X.reshape(n_samples, 1, n_features)
        X = torch.nn.functional.pad(X, (0, 300 - n_features), value=0)
        X = X.reshape(-1, 3, 10, 10)
        return X

    def fit(self, X, y, epochs: int = 30, batch_size: int = 1024, lr: float = 0.001):
        self.batch_size = batch_size
        X = self._process_input(X)
        y = torch.from_numpy(y).to(self.device).to(torch.float32)

        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        self.model.train()

        max_acc = 0
        best_model_path = Path(f"{self.model_name}_best.pkl")

        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0.0

            for i in range(0, len(X), batch_size):
                inputs = X[i : i + batch_size]
                labels = y[i : i + batch_size]

                optimizer.zero_grad()

                outputs = self.model(inputs).squeeze()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                predicted_labels = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                running_corrects += (predicted_labels == labels).sum()

            epoch_accuracy = running_corrects / X.shape[0]
            epoch_loss = running_loss / (X.shape[0] / batch_size)
            print(
                f"Epoch {epoch}/{epochs} - loss value: {epoch_loss} - accuracy: {epoch_accuracy}"
            )

            if epoch_accuracy > max_acc:
                max_acc = epoch_accuracy
                save_model_to_file(self, best_model_path)

        try:
            logger.info("Cleaning up CUDA memory")
            torch.cuda.empty_cache()
        except:
            pass

        if best_model_path.exists():
            self = load_model_from_file(best_model_path)
            best_model_path.unlink()

    def predict(self, X_test):
        X = self._process_input(X_test)

        self.model.eval()
        preds = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                inputs = X[i : i + self.batch_size]
                outputs = self.model(inputs).squeeze()

                y_preds = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                preds = torch.cat([preds, y_preds], dim=0)

        return preds.squeeze().cpu().numpy()
