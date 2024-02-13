# %%
import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display
from pandas.api.types import is_numeric_dtype

from tesco.constants import TESCO_COLORS
from tesco.data.preprocessing import load_preprocessed_data, load_raw_data

# %%

df = load_preprocessed_data("tesco_dataset")

# %%
os.makedirs("reports/figures/q1/", exist_ok=True)
# %%
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
        grouped = df.groupby(["is_train"])[col].value_counts(normalize=True).rename("Percentage").mul(100).reset_index()
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
plt.savefig("reports/figures/q1/feature_distributions.png", dpi=300)
# %%
# %%
df[~df["is_train"]]["location_id"].value_counts()
# %%
df["new_store"].dtype != "bool"

# %%
# Overlapping location_id or county

df[~df["is_train"]]["location_id"].isin(df[df["is_train"]]["location_id"].unique())
# %%
df[~df["is_train"]]["county"][~df[~df["is_train"]]["county"].isin(df[df["is_train"]]["county"].unique())]
# %%
df["county_number"] = df["county"].astype("int")
sns.scatterplot(data=df, x="county_number", y="normalised_sales", hue="is_train", size="is_train")
# %%
df["county_number"] = df["county"].astype("int")
count_correlated_features = (
    df[df.select_dtypes(include="number").columns].corr()["county_number"].abs().sort_values(ascending=False)
)
count_correlated_features.rename("Correlation with county_number", inplace=True)
Markdown(count_correlated_features.to_markdown())
# %%
fig, axis = plt.subplots(1, 3, figsize=(16 * 0.5, 9 * 0.5))
for ax, col in zip(axis.flatten(), list(count_correlated_features[1:4].index)):
    sns.scatterplot(
        data=df,
        x="county_number",
        y=col,
        hue="is_train",
        size="is_train",
        ax=ax,
        palette=[TESCO_COLORS["red"], TESCO_COLORS["blue"]],
    )
    if col == "household_affluency":
        ax.set_yscale("log")
        ax.set_title(f"Top 3 correlated features with county number")
    ax.set_xlabel("County number", fontsize=14)
    ax.set_ylabel(col.replace("_", " ").title(), fontsize=14)
plt.tight_layout()
plt.savefig("reports/figures/q1/top_3correlated_features.png", dpi=300)
# %%
df[~df["is_train"]]["county"].value_counts()
# %%
df[~df["is_train"]]["location_id"].value_counts()

# %%
sales_correlated_features = (
    df[df.select_dtypes(include="number").columns].corr()["normalised_sales"].abs().sort_values(ascending=False)
)
# %%
sales_correlated_features
# %%
df["location_id"].value_counts().max()
# %%
