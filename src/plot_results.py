import re
import pandas as pd
import matplotlib.pyplot as plt


def extract_data(df, col):
    regex = r"^([A-Za-z0-9\-]+)_([A-Za-z0-9\-]+)_(rot|no_rot) - ([A-Za-z0-9\-]+)$"
    matches = re.match(regex, col)
    if matches:
        dataset, model, rotation, metric = matches.groups()
        return dataset, model, rotation, metric
    else:
        return None


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

                for i, value in enumerate(result_df[col]):
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

    for i, (metric, metric_group) in enumerate(new_data_df.groupby("metric")):
        fig, axs = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
        axs = axs.ravel()

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

                boxplot_data.boxplot(
                    ax=ax, grid=False, showfliers=False, showmeans=True, meanline=True
                )

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"boxplots/dataset_{metric}.png")
        plt.close()


def plot_by_models():
    new_data_df, _ = parse_metric()

    for i, (metric, metric_group) in enumerate(new_data_df.groupby("metric")):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8), dpi=300)
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


if __name__ == "__main__":
    plot_by_metrics()
    plot_by_models()
