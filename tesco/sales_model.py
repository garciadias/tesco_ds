"""Develop a model to predict the normalised sales of a store for a given location."""

# %%
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tesco.constants import TESCO_COLORS
from tesco.data.preprocessing import load_preprocessed_data

# fix random seed for reproducibility
np.random.seed(1919)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns and nan values."""
    # Drop columns that are not useful for model training
    df = df.drop(
        columns=[
            "county",
        ]
    )
    df.set_index("location_id", inplace=True)
    return df


def split_data(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    # Split the data into train and test sets
    X_test = df[~df["is_train"]].drop(columns=["normalised_sales", "is_train"])  # noqa
    y_test = df[~df["is_train"]]["normalised_sales"]
    X_train, X_val, y_train, y_val = train_test_split(  # noqa
        df[df["is_train"]].drop(columns=["normalised_sales", "is_train"]),
        df[df["is_train"]]["normalised_sales"],
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16 * 0.7, 9 * 0.7))
    sns.scatterplot(
        data=df_original[df_original["is_train"].astype(bool)],
        x="household_affluency",
        y="normalised_sales",
        color="gray",
        label="Real values",
        alpha=0.5,
        ax=ax1,
    )
    sns.scatterplot(
        data=df_predicted,
        x="household_affluency",
        y="predicted_normalised_sales",
        hue="dataset",
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["yellow"], TESCO_COLORS["red"]],
        alpha=0.7,
        ax=ax1,
    )
    ax1.set_title("Model predictions Households affluency")
    sns.scatterplot(
        data=df_original[df_original["is_train"].astype(bool)],
        x="household_size",
        y="normalised_sales",
        color="gray",
        label="Real values",
        alpha=0.5,
        ax=ax2,
    )
    sns.scatterplot(
        data=df_predicted,
        x="household_size",
        y="predicted_normalised_sales",
        hue="dataset",
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["yellow"], TESCO_COLORS["red"]],
        alpha=0.7,
        ax=ax2,
    )
    ax2.set_title("Model predictions Household size")
    plt.tight_layout()
    return fig


def bland_altman_plot(df_predicted):
    # Bland–Altman plot
    df_bland_altman = df_predicted[["normalised_sales", "predicted_normalised_sales", "dataset"]].copy()
    df_bland_altman = df_bland_altman[df_bland_altman["dataset"].isin(["training", "validation"])]

    df_bland_altman["pred_real_mean"] = (
        df_bland_altman["normalised_sales"] + df_bland_altman["predicted_normalised_sales"]
    ) / 2
    df_bland_altman["pred_real_diff"] = (
        df_bland_altman["normalised_sales"] - df_bland_altman["predicted_normalised_sales"]
    )
    sns.scatterplot(
        data=df_bland_altman,
        x="pred_real_mean",
        y="pred_real_diff",
        hue="dataset",
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["red"]],
    )
    plt.axhline(0, color=TESCO_COLORS["yellow"], linestyle="--")
    plt.axhline(df_bland_altman["pred_real_diff"].mean(), color=TESCO_COLORS["green"], linestyle="--")
    plt.axhline(
        df_bland_altman["pred_real_diff"].mean() + 1.96 * df_bland_altman["pred_real_diff"].std(),
        color=TESCO_COLORS["green"],
        linestyle="--",
    )
    plt.axhline(
        df_bland_altman["pred_real_diff"].mean() - 1.96 * df_bland_altman["pred_real_diff"].std(),
        color=TESCO_COLORS["green"],
        linestyle="--",
    )
    plt.xlabel("Mean of predicted and real values")
    plt.ylabel("Difference between predicted and real values")
    plt.title("Bland–Altman plot")
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.ylim(-1 * max(np.abs(ylim)), max(np.abs(ylim)))
    plt.tight_layout()
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(
        xlim[0] + (xlim[1] - xlim[0]) * 0.05,
        ylim[1] - (ylim[1] - ylim[0]) * 0.1,
        f"Mean difference: {df_bland_altman['pred_real_diff'].mean():.3f}\n"
        f"± 1.96 * std: {1.96 * df_bland_altman['pred_real_diff'].std():.3f}",
        fontsize=12,
        color="black",
    )
    # move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt


# %%
if __name__ == "__main__":
    # %%
    random_state = 1919
    df = load_preprocessed_data("tesco_dataset")
    df = clean_data(df)
    # %%
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, random_state)
    # %%
    numeric_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns
    # %%
    model_pipeline = create_pipeline(RandomForestRegressor(), numeric_features, categorical_features)
    # Fit the model
    param_grid = {
        "Regressor__bootstrap": [True, False],
        "Regressor__criterion": ["squared_error"],
        "Regressor__max_depth": [2, 4, 7, 10, 20, None],
        "Regressor__max_features": ["log2", "sqrt", None, 1.0, 0.5],
        "Regressor__min_samples_leaf": [1, 2, 4],
        "Regressor__min_samples_split": [2, 5, 10],
        "Regressor__n_estimators": [10, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        "Regressor__random_state": [
            random_state,
            random_state - 10,
            random_state + 20,
            random_state + 3,
            random_state + 4,
        ],
    }
    regressor = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,
        n_iter=100,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=5,
        random_state=random_state,
        verbose=1,
    )
    regressor.fit(X_train, y_train)
    # %%
    baseline_regressor = create_pipeline(DummyRegressor(), numeric_features, categorical_features)
    baseline_regressor.fit(X_train, y_train)
    y_val_pred = regressor.predict(X_val)
    y_train_pred = regressor.predict(X_train)
    y_test_pred, final_model = test_set_prediction(X_train, X_val, X_test, y_train, y_val, regressor)

    y_pred_baseline = baseline_regressor.predict(X_val)
    y_train_pred_baseline = baseline_regressor.predict(X_train)
    y_test_pred_baseline = baseline_regressor.predict(X_test)

    df_complete = add_predictions_to_df(df, X_train, X_val, X_test, y_train_pred, y_val_pred, y_test_pred)
    df_complete_baseline = add_predictions_to_df(
        df, X_train, X_val, X_test, y_train_pred_baseline, y_pred_baseline, y_test_pred_baseline
    )
    model_metrics = metrics(df_complete, "normalised_sales", "predicted_normalised_sales")
    model_metrics_baseline = metrics(
        df_complete_baseline, "normalised_sales", "predicted_normalised_sales", "Dummy Regressor"
    )
    metrics_comparison = pd.concat([model_metrics, model_metrics_baseline], axis=1)
    display(metrics_comparison)

    # %%
    model_path = "models/tesco_dataset/"
    os.makedirs(model_path, exist_ok=True)
    model_file = "tesco_sales_model.pkl"
    results_file = "tesco_sales_model_results.csv"
    metrics_file = "tesco_sales_model_metrics.csv"
    metrics_comparison.to_csv(f"{model_path}{metrics_file}")

    test_set = df[~df["is_train"]].copy()
    test_set.loc[:, "normalised_sales"] = y_test_pred
    sales_quantiles = df["normalised_sales"].quantile([0.25, 0.5, 0.75])
    test_set["sales_quartile"] = pd.qcut(test_set["normalised_sales"], q=4, labels=["bottom_25%", "bottom_50%", "top_50%", "top_25%"])
    test_set.sort_values("normalised_sales", ascending=False, inplace=True)
    test_set.to_csv(f"{model_path}{results_file}")
    joblib.dump(final_model, f"{model_path}{model_file}")
    # %%
    fig = compare_original_and_predicted_data(df, df_complete)
    fig.savefig("models/tesco_dataset/tesco_sales_model_real_data_comparison.png")
    plt.close()
    fig = bland_altman_plot(df_complete)
    fig.savefig("models/tesco_dataset/tesco_sales_model_bland_altman_plot.png")
    plt.close()
    # %%
    # SHAP
    X_train_shap = final_model.named_steps["Preprocessor"].transform(X_train.dropna())
    model_predict = final_model.named_steps["Regressor"].predict
    explainer = shap.Explainer(model_predict, X_train_shap)
    shap_values_train = explainer(X_train_shap)
    # %%
    feature_names = (
        list(numeric_features) +
        list(final_model.named_steps["Preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features))
    )
    shap.summary_plot(shap_values_train, X_train_shap, feature_names=feature_names)
    plt.savefig("models/tesco_dataset/tesco_sales_model_shap_summary_plot.png", dpi=300)
