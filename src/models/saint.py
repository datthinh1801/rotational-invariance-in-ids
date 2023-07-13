import numpy as np
import torch
import ipdb

from pathlib import Path

import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from ipdb import set_trace
from sklearn.metrics import accuracy_score

from .saint_lib import SAINT, DataSetCatCon, embed_data_mask, data_prep_openml
from .utils import save_model_to_file, load_model_from_file


class SAINTModel:
    def __init__(self):
        self.model_name = "saint"
        self.cont_embeddings = "MLP"
        self.embedding_size = 32
        self.transformer_depth = 6
        self.attention_heads = 8
        self.attention_dropout = 0.1
        self.ff_dropout = 0.1
        self.attentiontype = "colrow"
        self.final_mlp_style = "sep"

        # default self.y_dim = 1 for single-label binary classification with BCEWithLogitsLoss
        self.y_dim = 1

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device in use: {self.device}")

    def embed_input(self, data):
        # x_categ is the the categorical data,
        # x_cont has continuous data,
        # y_gts has ground truth ys.
        # cat_mask is an array of ones same shape as x_categ and an additional column(corresponding to CLS token) set to 0s.
        # con_mask is an array of ones same shape as x_cont.
        x_categ, x_cont, y_gts, cat_mask, con_mask = (
            data[0].to(self.device),
            data[1].to(self.device),
            data[2].to(self.device),
            data[3].to(self.device),
            data[4].to(self.device),
        )

        # We are converting the data to embeddings in the next step
        _, x_categ_enc, x_cont_enc = embed_data_mask(
            x_categ, x_cont, cat_mask, con_mask, self.model
        )
        reps = self.model.transformer(x_categ_enc, x_cont_enc)

        # select only the representations corresponding to CLS token and apply mlp on it in the next step to get the predictions.
        y_reps = reps[:, 0, :]

        return y_reps, y_gts

    def fit(
        self,
        X_train,
        y_train,
        lr: float = 0.001,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        # preprocessing the dataset specifically for SAINT
        (
            cat_dims,
            cat_idxs,
            con_idxs,
            X_train,
            y_train,
            train_mean,
            train_std,
        ) = data_prep_openml(X_train, y_train)
        continuous_mean_std = np.array([train_mean, train_std]).astype(np.float32)

        # save to the object to use in inference later
        self.cat_idxs = cat_idxs
        self.batch_size = batch_size

        nfeat = X_train["data"].shape[1]
        if nfeat > 100:
            self.embedding_size = min(8, self.embedding_size)

        if self.attentiontype != "col":
            self.transformer_depth = 1
            self.attention_heads = min(4, self.attention_heads)
            self.attention_dropout = 0.8
            self.embedding_size = min(32, self.embedding_size)
            self.ff_dropout = 0.8

        train_ds = DataSetCatCon(X_train, y_train, cat_idxs, continuous_mean_std)
        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        n_classes = len(np.unique(y_train["data"][:, 0]))
        self.y_dim = 1 if n_classes == 2 else n_classes

        # Appending 1 for CLS token, this is later used to generate embeddings.
        cat_dims = np.append(np.array([1]), np.array(cat_dims)).astype(int)

        self.model = SAINT(
            categories=tuple(cat_dims),
            num_continuous=len(con_idxs),
            dim=self.embedding_size,
            dim_out=1,
            depth=self.transformer_depth,
            heads=self.attention_heads,
            attn_dropout=self.attention_dropout,
            ff_dropout=self.ff_dropout,
            mlp_hidden_mults=(4, 2),
            cont_embeddings=self.cont_embeddings,
            attentiontype=self.attentiontype,
            final_mlp_style=self.final_mlp_style,
            y_dim=self.y_dim,
        )
        self.model.to(self.device)

        criterion = nn.BCEWithLogitsLoss().to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        max_acc = 0
        best_model_path = Path(f"{self.model_name}_best.pkl")

        for epoch in range(epochs):
            self.model.train()

            running_loss = 0.0
            running_corrects = 0.0

            for data in trainloader:
                optimizer.zero_grad()

                y_reps, y_gts = self.embed_input(data)

                y_outs = self.model.mlpfory(y_reps).squeeze()
                y_gts = y_gts.to(torch.float32).squeeze()

                loss = criterion(y_outs, y_gts)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                predicted_labels = (torch.sigmoid(y_outs) >= 0.5).to(torch.float32)
                running_corrects += (predicted_labels == y_gts).sum()

            epoch_accuracy = running_corrects / len(trainloader.dataset)
            epoch_loss = running_loss / (len(trainloader.dataset) / batch_size)

            print(
                f"Epoch {epoch}/{epochs} - loss: {epoch_loss:.4f} - accuracy: {epoch_accuracy:.4f}"
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
        X = {"data": X_test, "mask": np.ones_like(X_test)}
        y_proxy = {"data": np.ones((X_test.shape[0], 1))}

        test_ds = DataSetCatCon(X, y_proxy, self.cat_idxs)
        testloader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        preds = torch.empty(0).to(self.device)

        with torch.no_grad():
            for data in testloader:
                y_reps, _ = self.embed_input(data)
                y_outs = self.model.mlpfory(y_reps)

                y_preds = (torch.sigmoid(y_outs) >= 0.5).to(torch.float32)
                preds = torch.cat([preds, y_preds], dim=0)

        return preds.squeeze().cpu().numpy()
