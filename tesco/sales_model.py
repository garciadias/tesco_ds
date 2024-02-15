"""Develop a model to predict the normalised sales of a store for a given location."""

# %%
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tesco.constants import TESCO_COLORS
from tesco.data.preprocessing import load_preprocessed_data
from tesco.utils import bland_altman_plot


def clean_data(df: pd.DataFrame, remove_columns=["county", "county_code"], index_col="location_id") -> pd.DataFrame:
    """Remove unwanted columns."""
    # Drop columns that are not useful for model training
    df = df.drop(columns=remove_columns)
    df.set_index(index_col, inplace=True)
    return df


def split_data(
    df: pd.DataFrame, random_state: int, y_var: str = "normalised_sales", dataset_var: str = "is_train"
) -> pd.DataFrame:
    """Split the data into train, validation and test sets."""
    X_test = df[~df[dataset_var]].drop(columns=[y_var, dataset_var])  # noqa
    y_test = df[~df[dataset_var]][y_var]
    X_train, X_val, y_train, y_val = train_test_split(  # noqa
        df[df[dataset_var]].drop(columns=[y_var, dataset_var]),
        df[df[dataset_var]][y_var],
        test_size=0.2,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_pipeline(core_model, numerical_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("MinMaxScaler Numerical scaler", MinMaxScaler(), numerical_features),
            ("Categoric encoder", OneHotEncoder(handle_unknown="infrequent_if_exist"), categorical_features),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("Preprocessor", preprocessor),
            ("Regressor", core_model),
        ]
    )
    return model_pipeline


def test_set_prediction(X_train, X_val, X_test, y_train, y_val, regressor):  # noqa
    final_model = regressor.best_estimator_
    X_train_all = pd.concat([X_train, X_val], ignore_index=True)  # noqa
    y_train_all = pd.concat([y_train, y_val], ignore_index=True)
    final_model.fit(X_train_all, y_train_all)
    y_pred_test = final_model.predict(X_test)
    return y_pred_test, final_model


def add_predictions_to_df(df, X_train, X_val, X_test, y_train_pred, y_val_pred, y_test_pred):  # noqa
    predictions_val = pd.DataFrame(
        {
            "location_id": X_val.index,
            "predicted_normalised_sales": y_val_pred,
            "dataset": "validation",
        }
    )
    predictions_train = pd.DataFrame(
        {
            "location_id": X_train.index,
            "predicted_normalised_sales": y_train_pred,
            "dataset": "training",
        }
    )
    predictions_test = pd.DataFrame(
        {
            "location_id": X_test.index,
            "predicted_normalised_sales": y_test_pred,
            "dataset": "test",
        }
    )
    predictions = pd.concat([predictions_val, predictions_train, predictions_test], ignore_index=True)
    df_complete = df.merge(predictions, on="location_id")
    return df_complete


def metrics(df_complete, original_column, predicted_column, model_name="Random Forest"):
    # Evaluate the model
    y_val = df_complete[df_complete["dataset"] == "validation"][original_column]
    y_val_pred = df_complete[df_complete["dataset"] == "validation"][predicted_column]
    y_train = df_complete[df_complete["dataset"] == "training"][original_column]
    y_train_pred = df_complete[df_complete["dataset"] == "training"][predicted_column]
    metrics = {
        f"Validation {model_name}": {
            "mean_squared_error": mean_squared_error(y_val, y_val_pred),
            "mean_absolute_error": mean_absolute_error(y_val, y_val_pred),
            "r2_score": r2_score(y_val, y_val_pred),
        },
        f"Train {model_name}": {
            "mean_squared_error": mean_squared_error(y_train, y_train_pred),
            "mean_absolute_error": mean_absolute_error(y_train, y_train_pred),
            "r2_score": r2_score(y_train, y_train_pred),
        },
    }
    metrics = pd.DataFrame(metrics)
    return metrics


def compare_original_and_predicted_data(df_original, df_predicted):
    # Plot the results
    # Scatter plot
    main_columns = ["household_affluency", "household_size", "crime_rate", "public_transport_dist"]
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 9))
    for ax, col in zip([ax1, ax2, ax3, ax4], main_columns):
        sns.scatterplot(
            data=df_original[df_original["is_train"].astype(bool)],
            x=col,
            y="normalised_sales",
            color="gray",
            label="Real values" if col == main_columns[0] else None,
            alpha=0.5,
            legend=False,
            ax=ax,
        )
        sns.scatterplot(
            data=df_predicted,
            x=col,
            y="predicted_normalised_sales",
            hue="dataset",
            palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["yellow"], TESCO_COLORS["red"]],
            alpha=0.7,
            ax=ax,
            legend=True if col == main_columns[0] else False,
        )
        col_title = col.replace("_", " ").title()
        ax.grid(True, which="major", linestyle="--")
        ax.set_xlabel(f"{col_title}", fontsize=16)
        if col == main_columns[0]:
            ax.set_ylabel("Normalised Sales", fontsize=16)
            ax.legend(fontsize=16)
        else:
            ax.set_ylabel("")
        for location_id in df_predicted[df_predicted["dataset"] == "test"]["location_id"]:
            ax.text(
                df_predicted.loc[df_predicted["location_id"] == location_id, col].values[0],
                df_predicted.loc[df_predicted["location_id"] == location_id, "predicted_normalised_sales"].values[0],
                location_id,
                fontsize=12,
            )
    fig.suptitle("Model predictions", fontsize=20)
    plt.tight_layout()
    return fig


def feature_importance_analysis(model, X_train):  # noqa
    X_train_shap = model.named_steps["Preprocessor"].transform(X_train.dropna())  # noqa
    model_predict = model.named_steps["Regressor"].predict
    explainer = shap.Explainer(model_predict, X_train_shap)
    shap_values_train = explainer(X_train_shap)
    numeric_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns
    feature_names = list(numeric_features) + list(
        model.named_steps["Preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
    )
    fig, ax = plt.subplots(1, 1, figsize=(16 * 0.7, 9 * 0.7))
    shap_values = pd.Series(shap_values_train.values.mean(axis=0), index=feature_names).sort_values(ascending=False)
    sns.barplot(x=shap_values, y=shap_values.index, ax=ax, color=TESCO_COLORS["blue"], orient="h")
    ax.vlines(0, -1, len(feature_names), color=TESCO_COLORS["red"], linestyle="--", linewidth=3.5)
    ax.set_title("Feature importance based on SHAP values", fontsize=20)
    ax.set_xlabel("Mean SHAP value", fontsize=16)
    ax.set_ylabel("Feature", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    # number bars ordering by absolute value
    abs_shap_values = shap_values.abs().sort_values(ascending=False)
    for i, (var_name, shap_value) in enumerate(shap_values.items()):
        position = [position + 1 for position, name in enumerate(abs_shap_values.index) if name == var_name][0]
        x_position = shap_value * 1.01 if shap_value > 0 else 0.0001
        ax.text(x_position, i, f"{position}", color="black", va="center", fontsize=14)
    plt.grid(True)
    plt.minorticks_on()
    plt.tight_layout()
    return fig
