import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from .utils import save_model_to_file, load_model_from_file


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.model_name = "mlp"
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class MLPModel:
    def __init__(self, hidden_size: int = 256):
        self.model_name = "mlp"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device in use: {self.device}")
        self.hidden_size = hidden_size

    def fit(
        self,
        X_train,
        y_train,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 0.001,
    ):
        input_size = X_train.shape[1]
        self.model = MLP(input_size=input_size, hidden_size=self.hidden_size).to(
            self.device
        )
        self.batch_size = batch_size

        X_train = torch.from_numpy(X_train).to(self.device).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(self.device).to(torch.float32)

        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        max_acc = 0
        best_model_path = Path(f"{self.model_name}_best.pkl")

        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            running_corrects = 0

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                # Prepare mini-batch
                inputs = X_train[i : i + batch_size]
                targets = y_train[i : i + batch_size]

                # Forward pass
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute statistics
                running_loss += loss.item()

                # calculate accuracy for the current batch
                predicted_labels = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                running_corrects += (predicted_labels == targets).sum()

            # Compute epoch statistics
            epoch_loss = running_loss / (len(X_train) / batch_size)
            epoch_acc = running_corrects / len(X_train)

            # Print loss and accuracy for every epoch
            print(
                f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_acc:.4f}"
            )

            if epoch_acc > max_acc:
                max_acc = epoch_acc
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
        X = torch.from_numpy(X_test).to(torch.float32).to(self.device)

        self.model.eval()
        preds = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                inputs = X[i : i + self.batch_size]
                outputs = self.model(inputs).squeeze()

                y_preds = (torch.sigmoid(outputs) > 0.5).to(torch.float32)
                preds = torch.cat([preds, y_preds], dim=0)

        return preds.cpu().numpy()
