import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.api.types import is_numeric_dtype

from tesco.constants import TESCO_COLORS
from tesco.data.preprocessing import load_preprocessed_data


def distribution_view(df: pd.DataFrame) -> plt.Figure:
    fig, axis = plt.subplots(4, 4, figsize=(16, 9))
    columns = list(df.columns)
    columns.remove("is_train")
    for ax, col in zip(axis.flatten(), columns):
        if is_numeric_dtype(df[col]) and df[col].dtype != "bool":
            sns.kdeplot(
                data=df,
                x=col,
                hue="is_train",
                common_norm=False,
                fill=True,
                ax=ax,
                alpha=0.7,
                palette=[TESCO_COLORS["red"], TESCO_COLORS["blue"]],
            )
        else:
            grouped = (
                df.groupby(["is_train"])[col].value_counts(normalize=True).rename("Percentage").mul(100).reset_index()
            )
            sns.barplot(
                data=grouped,
                x=col,
                y="Percentage",
                hue="is_train",
                ax=ax,
                palette=[TESCO_COLORS["red"], TESCO_COLORS["blue"]],
            )
            ax.tick_params(axis="x", rotation=0)
            if col in ["county", "location_id"]:
                n = len(ax.get_xticklabels()) // 10
                ax.set_xticklabels(ax.get_xticklabels()[::n])
                ax.set_xticks(ax.get_xticks()[::n])
        ax.set_title(col)
    plt.tight_layout()
    return fig


def top_correlated_features(df: pd.DataFrame, target: str, n_top: int = 3) -> plt.Figure:
    corr = df[df.select_dtypes(include="number").columns].corr()[target].abs().sort_values(ascending=False)
    target_title = target.replace("_", " ").title()
    corr.rename(f"Correlation with {target_title}", inplace=True)
    plot_cols = list(corr[1 : n_top + 1].index)
    fig, axis = plt.subplots(1, 3, figsize=(16 * 0.5, 9 * 0.5))

    for i, (ax, col) in enumerate(zip(axis.flatten(), list(plot_cols))):
        sns.scatterplot(
            data=df,
            x=target,
            y=col,
            hue="is_train",
            size="is_train",
            ax=ax,
            palette=[TESCO_COLORS["red"], TESCO_COLORS["blue"]],
        )
        if col == "household_affluency":
            ax.set_yscale("log")
        if i == n_top // 2:
            ax.set_title(f"Top {n_top} correlated features with {target_title}")
        ax.set_xlabel(target_title, fontsize=14)
        ax.set_ylabel(col.replace("_", " ").title(), fontsize=14)
    plt.tight_layout()
    return fig, corr


if __name__ == "__main__":
    output_dir = "reports/figures/q1"
    os.makedirs(output_dir, exist_ok=True)
    df = load_preprocessed_data("tesco_dataset")
    fig = distribution_view(df)

    plt.savefig(f"{output_dir}/feature_distributions.png", dpi=300)
