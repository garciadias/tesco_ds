"""Develop a model to predict the normalised sales of a store for a given location."""

# %%
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from tesco.constants import TESCO_COLORS
from tesco.data.preprocessing import load_preprocessed_data
from scipy.stats import randint


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unwanted columns."""
    # Drop columns that are not useful for model training
    df = df.drop(
        columns=[
            "county",
            "county_code",
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
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))
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
    ax1.set_title("Model predictions Households Affluency", fontsize=16)
    ax1.set_xlabel("Household Affluency", fontsize=16)
    ax1.set_ylabel("Normalised Sales", fontsize=16)
    ax1.legend(fontsize=16)
    # add labels to the test set points
    for location_id in df_predicted[df_predicted["dataset"] == "test"]["location_id"]:
        ax1.text(
            df_predicted.loc[df_predicted["location_id"] == location_id, "household_affluency"].values[0],
            df_predicted.loc[df_predicted["location_id"] == location_id, "predicted_normalised_sales"].values[0],
            location_id,
            fontsize=12,
        )
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
        legend=False,
    )
    ax2.set_title("Model predictions Household Size", fontsize=16)
    ax2.set_xlabel("Household Size", fontsize=16)
    ax2.set_ylabel("Normalised Sales", fontsize=16)
    for location_id in df_predicted[df_predicted["dataset"] == "test"]["location_id"]:
        ax2.text(
            df_predicted.loc[df_predicted["location_id"] == location_id, "household_size"].values[0],
            df_predicted.loc[df_predicted["location_id"] == location_id, "predicted_normalised_sales"].values[0],
            location_id,
            fontsize=12,
        )
    sns.scatterplot(
        data=df_original[df_original["is_train"].astype(bool)],
        x="crime_rate",
        y="normalised_sales",
        color="gray",
        label="Real values",
        alpha=0.5,
        ax=ax3,
    )
    sns.scatterplot(
        data=df_predicted,
        x="crime_rate",
        y="predicted_normalised_sales",
        hue="dataset",
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["yellow"], TESCO_COLORS["red"]],
        alpha=0.7,
        ax=ax3,
        legend=False,
    )
    ax3.set_title("Model predictions Crime Rate", fontsize=16)
    ax3.set_xlabel("Crime Rate", fontsize=16)
    ax3.set_ylabel("Normalised Sales", fontsize=16)
    for location_id in df_predicted[df_predicted["dataset"] == "test"]["location_id"]:
        ax3.text(
            df_predicted.loc[df_predicted["location_id"] == location_id, "crime_rate"].values[0],
            df_predicted.loc[df_predicted["location_id"] == location_id, "predicted_normalised_sales"].values[0],
            location_id,
            fontsize=12,
        )
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
    df_bland_altman_validation = df_bland_altman[df_bland_altman["dataset"] == "validation"]
    fig, ax = plt.subplots(1, 1, figsize=(16 * 0.7, 9 * 0.7))
    sns.scatterplot(
        data=df_bland_altman,
        x="pred_real_mean",
        y="pred_real_diff",
        hue="dataset",
        palette=[TESCO_COLORS["light_blue"], TESCO_COLORS["red"]],
        ax=ax,
    )
    ax.axhline(0, color=TESCO_COLORS["yellow"], linestyle="-")
    ax.axhline(df_bland_altman_validation["pred_real_diff"].mean(), color=TESCO_COLORS["green"], linestyle="--")
    ax.axhline(
        df_bland_altman_validation["pred_real_diff"].mean() + 1.96 * df_bland_altman_validation["pred_real_diff"].std(),
        color=TESCO_COLORS["green"],
        linestyle="--",
    )
    ax.axhline(
        df_bland_altman_validation["pred_real_diff"].mean() - 1.96 * df_bland_altman_validation["pred_real_diff"].std(),
        color=TESCO_COLORS["green"],
        linestyle="--",
    )
    ax.set_xlabel("Mean of predicted and real values", fontsize=20)
    ax.set_ylabel("Real - Predicted values", fontsize=20)
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
        f"Validation Mean difference: {df_bland_altman_validation['pred_real_diff'].mean():.3f}\n"
        f"± 1.96 * std: {1.96 * df_bland_altman_validation['pred_real_diff'].std():.3f}",
        fontsize=16,
        color="black",
    )
    # move legend outside of the plot
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=16)
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


# %%
if __name__ == "__main__":
    # %%
    df = load_preprocessed_data("tesco_dataset")
    df["county_code"] = df["county"].astype("int")
    # Remove the county column and set location_id as the Index
    Xy = clean_data(df.copy())
    # Split the data for Training, validating and testing accordingly
    random_state = 1919
    # fix random seed for reproducibility
    np.random.seed(random_state)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(Xy, random_state)
    # Create a pipeline to handle preprocessing and model training
    # The pipeline applies Standard Scaling of the numerical variables and One Hot Encoding of the categorical variables
    numeric_features = X_train.select_dtypes(include=["number"]).columns
    categorical_features = X_train.select_dtypes(include=["object", "string"]).columns
    # The pipeline is flexible to the use of different models. In this case, we are sticking to the requirement of using
    # Random Forest
    model_pipeline = create_pipeline(RandomForestRegressor(), numeric_features, categorical_features)
    # %%
    # Use RandomizedSearchCV to optimise the hyperparameters of the model
    param_grid = {
        "Regressor__n_estimators": [80, 100, 120, 200],
        "Regressor__random_state": randint(random_state - 10, random_state + 10),
    }
    regressor = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_grid,
        n_iter=100,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=2,
        random_state=random_state,
        verbose=1,
    )
    regressor.fit(X_train, y_train)
    # %%
    # Train a baseline model for comparison. I have compared it with a Dummy Regressor, as a Linear Regressor,
    # but here I will use a non-optimised model as a comparison.
    baseline_regressor = create_pipeline(RandomForestRegressor(random_state=random_state), numeric_features,
                                         categorical_features)
    baseline_regressor.fit(X_train, y_train)
    # %%
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
        df_complete_baseline, "normalised_sales", "predicted_normalised_sales", "Vanilla Random Forest Regressor"
    )
    metrics_comparison = pd.concat([model_metrics, model_metrics_baseline], axis=1)
    print("# Creating Table 2: Model Performance Metrics")
    print(metrics_comparison)
    # %%
    model_path = "models/tesco_dataset/"
    os.makedirs(model_path, exist_ok=True)
    # %%
    print("# Creating Figure 4a: Visual inspection of predictions")
    fig = bland_altman_plot(df_complete)
    fig.savefig("models/tesco_dataset/tesco_sales_model_bland_altman_plot.png")
    plt.close()
    print("Bland-Altman plot saved saved at models/tesco_dataset/tesco_sales_model_bland_altman_plot.png")
    print("# Creating Figure 4b")
    fig = compare_original_and_predicted_data(df, df_complete)
    fig.savefig("models/tesco_dataset/tesco_sales_model_real_data_comparison.png")
    plt.show()
    print("Real data comparison saved at models/tesco_dataset/tesco_sales_model_real_data_comparison.png")
    # %%
    # Feature importance analysis
    print("# Creating Figure 5")
    feature_importance_analysis(final_model, X_train)
    plt.close()
    print("Feature importance analysis saved at models/tesco_dataset/tesco_sales_model_feature_importance.png")
    # %%
    # Save model results
    model_file = "tesco_pkl"
    results_file = "tesco_sales_model_results.csv"
    metrics_file = "tesco_sales_model_metrics.csv"
    metrics_comparison.to_csv(f"{model_path}{metrics_file}")

    test_set = df[~df["is_train"]].copy()
    test_set.loc[:, "normalised_sales"] = y_test_pred
    sales_quantiles = df["normalised_sales"].quantile([0.25, 0.5, 0.75])
    test_set["sales_quartile"] = pd.qcut(
        test_set["normalised_sales"],
        q=100,
        labels=[f"top {100 - percent}%" if percent > 50 else f"bottom {percent}%" for percent in range(0, 100, 1)],
    )
    test_set.sort_values("normalised_sales", ascending=False, inplace=True)
    test_set.to_csv(f"{model_path}{results_file}")
    joblib.dump(final_model, f"{model_path}{model_file}")
    print("# Create Table 3: Predicted sales")
    print(test_set.set_index("location_id")[["normalised_sales", "sales_quartile"]].sort_values(
        "normalised_sales", ascending=False
    ))
