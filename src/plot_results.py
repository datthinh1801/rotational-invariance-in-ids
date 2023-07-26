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


if __name__ == "__main__":
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
    new_data_df.to_csv("results/final_results.csv", index=False)

    for metric in metrics:
        grouped_data = new_data_df[new_data_df["metric"] == metric].groupby("rotation")

        fig, axs = plt.subplots(2, 5, figsize=(20, 8), dpi=300)
        # fig.suptitle(f"Metric: {metric}", fontsize=16)
        axs = axs.ravel()

        for i, (rotation, group) in enumerate(grouped_data):
            for j, (dataset, sub_group) in enumerate(group.groupby("dataset")):
                boxplot_data = sub_group.reset_index().pivot(
                    index="iteration", columns="model", values="value"
                )

                ax = axs[i * 5 + j]
                ax.set_title(f"Dataset: {dataset}")
                ax.set_xlabel("Model")
                ax.set_ylabel(f"{rotation} {metric}")

                boxplot_data.boxplot(
                    ax=ax, grid=False, showfliers=True, showmeans=True, meanline=True
                )
                ax.set_xticklabels(boxplot_data.columns)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(f"boxplots/metrics_{metric}.png")
        plt.close()
