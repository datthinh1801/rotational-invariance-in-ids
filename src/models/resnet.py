import ipdb

import timm
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score


class ResNet:
    def __init__(
        self,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use: {self.device}")

        # num_classes is set to 1 to be compatible with the BCEWithLogitLoss
        # which requires num_classes=1 for single-label binary classification
        self.model = timm.create_model("resnet18", pretrained=False, num_classes=1).to(
            self.device
        )

    def fit(self, X, y, epochs: int = 30, batch_size: int = 1024, lr: float = 0.001):
        self.batch_size = batch_size

        n_samples, n_features = X.shape
        X = torch.from_numpy(X).to(self.device).to(torch.float32)
        X = X.reshape(n_samples, 1, n_features)
        X = torch.nn.functional.pad(X, (0, 300 - n_features), value=0)
        X = X.reshape(-1, 3, 10, 10)

        y = torch.from_numpy(y).to(self.device).to(torch.float32)

        # Define the loss function and optimizer
        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            batch_accuracy = 0.0

            for i in range(0, len(X), batch_size):
                inputs = X[i : i + batch_size]
                labels = y[i : i + batch_size]

                optimizer.zero_grad()

                outputs = self.model(inputs).squeeze()

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                # Calculate accuracy for the current batch
                predicted_labels = (
                    torch.sigmoid(outputs) > 0.5
                ).float()  # Apply sigmoid and threshold for binary prediction
                batch_accuracy = (predicted_labels == labels).sum().item() / len(labels)

                running_loss += loss.item()
                running_accuracy += batch_accuracy

            epoch_accuracy = running_accuracy / (len(X) / batch_size)
            epoch_loss = running_loss / (len(X) / batch_size)
            print(
                f"Epoch: {epoch} - loss value: {epoch_loss} - accuracy: {epoch_accuracy}"
            )

        try:
            logger.info("Cleaning up CUDA memory")
            torch.cuda.empty_cache()
        except:
            pass

    def predict(self, X_test):
        n_samples, n_features = X_test.shape
        X = torch.from_numpy(X_test).to(torch.float32).to(self.device)
        X = X.reshape(n_samples, 1, n_features)
        X = torch.nn.functional.pad(X, (0, 300 - n_features), value=0)
        X = X.reshape(-1, 3, 10, 10)

        self.model.eval()
        preds = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                inputs = X[i : i + self.batch_size]
                outputs = self.model(input).squeeze()

                preds = torch.cat([preds, torch.argmax(outputs, dim=1).float()], dim=0)

        return preds.cpu().numpy()
