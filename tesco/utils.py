import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tesco.constants import TESCO_COLORS


def bland_altman_plot(df: pd.DataFrame, target: str = "normalised_sales", dataset_var: str = "dataset", remove_test: bool = False) -> plt.Figure:
    """Create a Bland–Altman plot to compare the agreement between the predicted and the real data.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the predicted and real values to compare.
    target : str, optional
        The target variable to compare, by default "normalised_sales"
    dataset_var : str, optional
        The variable containing the dataset information, by default "dataset". This variable is used to color the plot
        and to calculate the mean difference and the standard deviation. The training dataset is not used to calculate
        the mean difference and the standard deviation and is shown in a different color to the validation dataset.
        Systematic differences between the behavior of the training and validation datasets can be identified.

    Returns
    -------
    plt.Figure
        The Bland–Altman plot.
    """
    # Bland–Altman plot
    if remove_test:
        df = df[df[dataset_var] != "test"].copy()

    df.loc[:, "pred_real_mean"] = (df[f"predicted_{target}"] + df[target]) / 2
    df.loc[:, "pred_real_diff"] = df[f"predicted_{target}"] - df[target]

    fig, ax = plt.subplots(1, 1, figsize=(16 * 0.7, 9 * 0.7))
    sns.scatterplot(
        data=df,
        x="pred_real_mean",
        y="pred_real_diff",
        hue=dataset_var,
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["red"]],
        ax=ax,
    )
    mean_diff = df[df[dataset_var] != "training"]["pred_real_diff"].mean()
    std_diff = df[df[dataset_var] != "training"]["pred_real_diff"].std()
    ax.axhline(0, color=TESCO_COLORS["green"], linestyle="-")
    ax.axhline(mean_diff, color=TESCO_COLORS["red"], linestyle="--")
    ax.axhline(
        mean_diff + 1.96 * std_diff,
        color=TESCO_COLORS["yellow"],
        linestyle="--",
    )
    ax.axhline(
        mean_diff - 1.96 * std_diff,
        color=TESCO_COLORS["yellow"],
        linestyle="--",
    )
    ax.set_xlabel("Mean of predicted and real values", fontsize=20)
    ax.set_ylabel("Predicted values - Real", fontsize=20)
    ax.set_title("Bland–Altman plot", fontsize=20)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.ylim(-1 * max(np.abs(ylim)), max(np.abs(ylim)))
    plt.tight_layout()
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(
        xlim[0] + (xlim[1] - xlim[0]) * 0.05,
        ylim[1] - (ylim[1] - ylim[0]) * 0.1,
        f"Validation Mean difference: {mean_diff:.3f}\n" f"± 1.96 * std: {1.96 * std_diff:.3f}",
        fontsize=16,
        color="black",
    )
    # move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=16)
    plt.tight_layout()
    return fig
