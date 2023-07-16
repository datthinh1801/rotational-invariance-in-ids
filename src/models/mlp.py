import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path

from sklearn.model_selection import train_test_split

from .utils import save_model_to_file, load_model_from_file


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model_name = "mlp"
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

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
        self.model = MLP(
            input_size=input_size, hidden_size=self.hidden_size, output_size=1
        ).to(self.device)
        self.batch_size = batch_size
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15)

        X_train = torch.from_numpy(X_train).to(self.device).to(torch.float32)
        y_train = torch.from_numpy(y_train).to(self.device).to(torch.float32)
        X_valid = torch.from_numpy(X_valid).to(self.device).to(torch.float32)
        y_valid = torch.from_numpy(y_valid).to(self.device).to(torch.float32)

        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=lr)

        best_valid_acc = 0.0
        best_model_path = Path(f"{self.model_name}_temp.pkl")

        self.model.train()
        for epoch in range(epochs):
            running_loss = []

            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                # Prepare mini-batch
                inputs = X_train[i: i + batch_size]
                targets = y_train[i: i + batch_size]

                # Forward pass
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Compute statistics
                running_loss.append(loss.detach().item())

            self.model.eval()
            valid_corrects = 0.0
            for i in range(0, len(X_valid), batch_size):
                inputs = X_valid[i: i + batch_size]
                targets = y_valid[i: i + batch_size]

                outputs = self.model(inputs).squeeze()
                preds = (torch.sigmoid(targets) >= 0.5).float()

                valid_corrects += (preds == outputs).sum()

            # Compute epoch statistics
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

    def predict(self, X):
        X = torch.from_numpy(X).to(torch.float32).to(self.device)

        self.model.eval()
        preds = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                inputs = X[i: i + self.batch_size]
                outputs = self.model(inputs).squeeze()

                preds = torch.cat(
                    [preds, (torch.sigmoid(outputs) >= 0.5).to(torch.float32)], dim=0
                )

        return preds.cpu().numpy()
