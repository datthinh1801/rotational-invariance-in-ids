import ipdb

import timm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

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
        self.num_classes = 1
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=self.num_classes
        ).to(self.device)

    def _process_input(self, X):
        n_samples, n_features = X.shape

        x = int(np.ceil(np.sqrt(n_features / 3)))
        target_n_features = 3 * x * x

        if n_features < target_n_features:
            padding = target_n_features - n_features
            X = np.pad(X, [(0, 0), (0, padding)], mode="constant")

        X = X.reshape((n_samples, 3, x, x))
        X = torch.from_numpy(X).to(self.device).to(torch.float32)
        return X

    def fit(self, X, y, epochs: int = 30, batch_size: int = 1024, lr: float = 0.001):
        self.batch_size = batch_size
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15)

        X_train = self._process_input(X_train)
        y_train = torch.from_numpy(y_train).to(self.device).float()
        X_valid = self._process_input(X_valid)
        y_valid = torch.from_numpy(y_valid).to(self.device).float()

        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        best_valid_acc = 0.0
        best_model_path = Path(f"{self.model_name}_temp.pkl")

        self.model.train()
        for epoch in range(epochs):
            running_loss = []

            for i in range(0, len(X_train), batch_size):
                inputs = X_train[i: i + batch_size]
                labels = y_train[i: i + batch_size]

                optimizer.zero_grad()

                outputs = self.model(inputs).squeeze()

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.detach().item())

            self.model.eval()
            valid_corrects = 0.0
            with torch.no_grad():
                for i in range(0, len(X_valid), batch_size):
                    inputs = X_valid[i: i + batch_size]
                    labels = y_valid[i: i + batch_size]
                    outputs = self.model(inputs).squeeze()
                    preds = (torch.sigmoid(outputs) >= 0.5).float()
                    valid_corrects += (preds == labels).sum()

            epoch_loss = torch.mean(torch.tensor(running_loss).to(self.device))
            epoch_valid_accuracy = valid_corrects / len(X_valid)
            print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.4f} - valid accuracy: {epoch_valid_accuracy:.4f}")

            if epoch_valid_accuracy > best_valid_acc:
                best_valid_acc = epoch_valid_accuracy
                print("Saving checkpoint")
                save_model_to_file(self, best_model_path)

        try:
            print("Cleaning up CUDA memory")
            torch.cuda.empty_cache()
        except:
            pass

        if best_model_path.exists():
            print("Loading best model from checkpoint")
            self = load_model_from_file(best_model_path)
            best_model_path.unlink()

    def predict(self, X_test):
        X = self._process_input(X_test)

        self.model.eval()
        preds = torch.empty(0).to(self.device)
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                inputs = X[i: i + self.batch_size]
                outputs = self.model(inputs).squeeze()

                preds = torch.cat(
                    [preds, (torch.sigmoid(outputs) >= 0.5).to(torch.float32)], dim=0
                )

        return preds.squeeze().cpu().numpy()
