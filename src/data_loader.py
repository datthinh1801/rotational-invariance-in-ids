import pandas as pd
import numpy as np

from pathlib import Path

from configs.global_config import GLOBAL_CONFIG
from configs.data_loader_config import DATALOADER_CONFIG
from configs.data_config import DATA_CONFIG


class DataLoader:
    def __init__(
        self,
        data_path: Path,
        target_col: str,
        chunk_size: int = None,
        frac: float = None,
        n_max: int = None,
        seed=None,
    ) -> None:
        """Initialization.

        Args:
            data_path (Path): Path to the dataset.
            target_col (str): The name of the target column.
            chunk_size (int, optional): Size of the chunk. Defaults to 10000.
            frac (float, optional): Fraction of data to load (from 0 to 1). Defaults to 1.
            max_n (int, optional): The max number of samples to load. Defaults to -1 (for loading all samples).
            seed (_type_, optional): Random seed. Defaults to None.
        """
        # initialization parameters
        self.data_path = data_path
        self.target_col = target_col
        self.feats = None
        self.frac = frac or DATALOADER_CONFIG["frac"]
        self.n_max = n_max or DATALOADER_CONFIG["n_max"]
        self.seed = seed
        self.label_cnts = pd.Series(dtype=float)

        # chunk parameters
        self.current_chunk = 0
        self.chunk_size = chunk_size or DATALOADER_CONFIG["chunk_size"]
        self.last_chunk = pd.DataFrame()

    def load_data(
        self,
    ) -> (pd.DataFrame, pd.Series):
        """A helper function to load the dataset at a single call instead of loading by chunks.

        Returns:
            pd.DataFrame: The dataset in the pd.DataFrame format
        """
        X = pd.DataFrame()
        y = pd.Series()

        if 0 < self.frac < 1:
            for X_chunk, y_chunk in self:
                X = self._concat_data(X, X_chunk)
                y = self._concat_data(y, y_chunk)
        elif self.n_max > 0:
            for X_chunk, y_chunk in self:
                X = X_chunk
                y = y_chunk
        return X, y

    def _load_next_chunk(
        self,
    ) -> (pd.DataFrame, pd.Series):
        """Load the next data chunk."""
        chunk_start = self.current_chunk * self.chunk_size
        try:
            chunk = pd.read_csv(
                self.data_path,
                skiprows=range(chunk_start),
                nrows=self.chunk_size,
                names=self.feats,
            )
        except UnicodeDecodeError:
            chunk = pd.read_csv(
                self.data_path,
                skiprows=range(chunk_start),
                nrows=self.chunk_size,
                names=self.feats,
                encoding="cp1252",
            )
        except Exception as e:
            print(e)
            exit(1)

        if chunk.shape[0] == 0:
            raise StopIteration

        if self.feats is None:
            chunk.columns = chunk.columns.str.strip()
            self.feats = chunk.columns

        if 0 < self.frac < 1:
            chunk = self._load_chunk_fraction(chunk)
        elif self.n_max > 0:
            chunk = self._load_chunk_n(chunk)

        try:
            y_chunk = chunk[self.target_col]
            X_chunk = chunk.drop(columns=[self.target_col])
            self.current_chunk += 1
            return X_chunk, y_chunk
        except:
            return None, None

    def _load_chunk_fraction(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Load a fraction of each label in the current chunk.

        Args:
            chunk (pd.DataFrame): A data chunk.
        """
        unique_target_labels = chunk[self.target_col].unique()
        final_chunk = pd.DataFrame()
        for label in unique_target_labels:
            label_samples = chunk[chunk[self.target_col] == label]
            fragmented_chunk = label_samples.sample(
                frac=self.frac, random_state=self.seed, axis=0, ignore_index=True
            )
            final_chunk = self._concat_data(final_chunk, fragmented_chunk)
        return final_chunk

    def _load_chunk_n(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Load max_n number of samples while maintaining the data distribution of the target column.

        Args:
            chunk (pd.DataFrame): A data chunk.
        """
        new_data = self._concat_data(self.last_chunk, chunk)
        if new_data.shape[0] <= self.n_max:
            self.last_chunk = new_data
            return new_data

        new_data_label_cnts = new_data[self.target_col].value_counts()
        self.label_cnts = self.label_cnts.add(new_data_label_cnts, fill_value=0)
        props_dist = self._calculate_probability_distribution(self.label_cnts)

        final_chunk = pd.DataFrame()
        for label in props_dist.index:
            props = props_dist[label]
            samples = new_data[new_data[self.target_col] == label]
            n_samples = int(np.min([np.ceil(self.n_max * props), samples.shape[0]]))
            fragmented_chunk = samples.sample(
                n=n_samples, random_state=self.seed, axis=0, ignore_index=True
            )
            final_chunk = self._concat_data(final_chunk, fragmented_chunk)
        self.last_chunk = final_chunk
        return final_chunk

    @staticmethod
    def _concat_data(data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:
        """Concatenate 2 DataFrames.

        Args:
            data1 (pd.DataFrame): DataFrame 1
            data2 (pd.DataFrame): DataFrame 2

        Returns:
            pd.DataFrame: The concatenated DataFrame
        """
        new_data = pd.concat([data1, data2], axis=0, ignore_index=True).reset_index(
            drop=True
        )
        return new_data

    @staticmethod
    def _calculate_probability_distribution(series: pd.Series) -> pd.Series:
        """Calculate the probability distribution of the given Series.

        Args:
            series (pd.Series): The Series to calculate the probability distribution from.

        Returns:
            pd.Series: A Series of the probability distribution
        """
        total_count = series.sum()
        probabilities = series / total_count
        return probabilities

    def __iter__(self):
        return self

    def __next__(self):
        X_chunk, y_chunk = self._load_next_chunk()
        if X_chunk.shape[0] == 0:
            raise StopIteration
        return X_chunk, y_chunk


if __name__ == "__main__":
    data_path = Path(".") / "data" / "IoTID20_final.csv"
    data_loader = DataLoader(
        data_path=data_path, target_col="Label", frac=0.1, seed=42, chunk_size=100000
    )
    X = pd.DataFrame()
    y = pd.Series()
    for X_chunk, y_chunk in data_loader:
        X = data_loader._concat_data(X, X_chunk)
        y = data_loader._concat_data(y, y_chunk)
    print("Read 10% by chunk")
    print(X.shape)
    print(data_loader._calculate_probability_distribution(y))
    print("Read 10% at once")
    fulldata_loader = DataLoader(
        data_path=data_path, target_col="Label", frac=0.1, seed=42
    )
    X, y = fulldata_loader.load_data()
    print(X.shape)
    print(fulldata_loader._calculate_probability_distribution(y))
