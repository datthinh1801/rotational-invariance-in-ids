import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def extract_data(df, col):
    regex = r"^([A-Za-z0-9\-]+)_([A-Za-z0-9\-]+)_(rot|no_rot) - ([A-Za-z0-9\-]+)$"
    matches = re.match(regex, col)
    if matches:
        dataset, model, rotation, metric = matches.groups()
        return dataset, model, rotation, metric
    else:
        return None


def normalize_metric(acc, min_value, max_value):
    """
    Normalize accuracy between 0 and 1 using an affine renormalization.
    """
    return (acc - min_value) / (max_value - min_value)


def remove_outliers_iqr(data):
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    # Calculate the IQR
    iqr = q3 - q1

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return lower_bound, upper_bound


def parse_metric():
    metrics = ["accuracy", "f1-score", "precision", "recall"]
    new_data = {
        "iteration": [],
        "dataset": [],
        "model": [],
        "rotation": [],
        "metric": [],
        "value": [],
    }

    for metric in metrics:
        result_df = pd.read_csv(f"results/{metric}.csv")

        for col in result_df.columns:
            col_info = extract_data(result_df, col)

            if col_info:
                dataset, model, rotation, metric = col_info
                iterations = range(1, len(result_df) + 1)

                # Calculate outlier boundaries for the current set of (metric, dataset, rotation, model)
                lower_bound, upper_bound = remove_outliers_iqr(result_df[col])

                for i, value in enumerate(result_df[col]):
                    # Check if the value is within the outlier boundaries
                    if lower_bound <= value <= upper_bound:
                        new_data["iteration"].append(iterations[i])
                        new_data["dataset"].append(dataset)
                        new_data["model"].append(model)
                        new_data["rotation"].append(rotation)
                        new_data["metric"].append(metric)
                        new_data["value"].append(value)

    new_data_df = pd.DataFrame(new_data)

    return new_data_df, metrics


def plot_by_metrics():
    new_data_df, metrics = parse_metric()
    lower_bound = new_data_df.groupby(["metric", "dataset"])["value"].min()

    for i, (metric, metric_group) in enumerate(new_data_df.groupby("metric")):
        n_rows, n_cols, plot_size = (2, 5, 4)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size), dpi=300
        )
        axs = axs.ravel()

        # Create an empty DataFrame to store the statistics for all rotations of the dataset
        all_rotation_stats = pd.DataFrame()

        for j, (rotation, rotation_group) in enumerate(
            metric_group.groupby("rotation")
        ):
            for k, (dataset, dataset_group) in enumerate(
                rotation_group.groupby("dataset")
            ):
                boxplot_data = dataset_group.reset_index().pivot(
                    index="iteration", columns="model", values="value"
                )

                ax = axs[j * 5 + k]
                ax.set_title(f"Dataset: {dataset}")
                ax.set_xlabel("Model")
                ax.set_ylabel(f"{rotation} {metric}")

                y_margin_percent = 5
                y_min = lower_bound.loc[(metric, dataset)]
                y_max = 1
                y_range = y_max - y_min
                y_margin = y_range * y_margin_percent / 100

                # Set the y-axis limits with the margin
                ax.set_ylim(y_min - y_margin, y_max + y_margin)
                # ax.set_ylim(lower_bound.loc[(metric, dataset)], 1)

                boxplot_data.boxplot(
                    ax=ax, grid=False, showfliers=False, showmeans=True, meanline=True
                )

                # Calculate statistics (median, mean, etc.) and store them in a DataFrame
                boxplot_stats = boxplot_data.describe().transpose()
                boxplot_stats.reset_index(inplace=True)
                boxplot_stats.rename(columns={"index": "Model"}, inplace=True)

                # Add a new column for the rotation
                boxplot_stats["Rotation"] = rotation
                boxplot_stats["Dataset"] = dataset

                # Append the statistics to the DataFrame containing all rotations of the dataset
                all_rotation_stats = pd.concat([all_rotation_stats, boxplot_stats])

        # Export combined statistics to a CSV file
        output_filename = f"boxplots/{metric}_stats.csv"
        all_rotation_stats.to_csv(output_filename, index=False)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"boxplots/details_{metric}.png")
        plt.close()


def plot_by_models():
    new_data_df, _ = parse_metric()

    for i, (metric, metric_group) in enumerate(new_data_df.groupby("metric")):
        n_rows, n_cols, plot_size = (2, 4, 4)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size), dpi=300
        )
        axs = axs.ravel()

        for j, (model, model_group) in enumerate(metric_group.groupby("model")):
            ax = axs[j]
            for rotation, rotation_group in model_group.groupby("rotation"):
                boxplot_data = rotation_group["value"]

                ax.boxplot(
                    boxplot_data,
                    positions=[int(rotation == "rot")],
                    showfliers=False,
                    showmeans=True,
                    meanline=True,
                )
            ax.set_title(f"Model: {model}")
            ax.set_xlabel("Rotation")
            ax.set_ylabel(metric)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"boxplots/models_{metric}")
        plt.close()


def plot_by_datasets():
    new_data_df, _ = parse_metric()

    for i, (metric, metric_group) in enumerate(new_data_df.groupby("metric")):
        n_rows, n_cols, plot_size = (1, 5, 4)
        fig, axs = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * plot_size, n_rows * plot_size), dpi=300
        )
        axs = axs.ravel()

        for j, (dataset, dataset_group) in enumerate(metric_group.groupby("dataset")):
            ax = axs[j]
            for rotation, rotation_group in dataset_group.groupby("rotation"):
                boxplot_data = rotation_group["value"]

                ax.boxplot(
                    boxplot_data,
                    positions=[int(rotation == "rot")],
                    showfliers=False,
                    showmeans=True,
                    meanline=True,
                )
            ax.set_title(f"Dataset: {dataset}")
            ax.set_xlabel("Rotation")
            ax.set_ylabel(metric)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"boxplots/datasets_{metric}")
        plt.close()


if __name__ == "__main__":
    plot_by_metrics()
    plot_by_models()
    plot_by_datasets()
