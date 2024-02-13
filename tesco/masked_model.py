# %%
from typing import Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from tesco.data.preprocessing import load_preprocessed_data
from tesco.constants import TESCO_COLORS


@dataclass
class ModelSelector:
    models: list[object]
    model_name: str
    decomposition: Union[None, object]
    scaler: Union[None, object]

    def create_model_pipeline(self):
        self.pipelines = []
        for model in self.models:
            model_steps = []
            if self.scaler:
                model_steps.append(("scaler", self.scaler))
            if self.decomposition:
                model_steps.append(("decomposition", self.decomposition))
            model_steps.append(("model", model))

            model_pipeline = Pipeline(
                steps=model_steps
            )
            self.pipelines.append(model_pipeline)


    def fit(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]): # noqa
        self.create_model_pipeline()
        self.fitted_models = {}
        for model, pipeline in zip(self.models, self.pipelines):
            pipeline.fit(X_train, y_train)
            self.fitted_models[model.__class__.__name__] = pipeline

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]): # noqa
        self.predictions_ = {}
        for pipeline_name, pipeline in self.fitted_models.items():
            self.predictions_[pipeline_name] = pipeline.predict(X_test)

    def evaluate(self, y_true: Union[np.ndarray, pd.Series]):
        self.evaluation_ = {}
        for model_name, predictions in self.predictions_.items():
            if self.decomposition:
                model_name = f"{self.decomposition.__class__.__name__}_{model_name}"
            if self.scaler:
                model_name = f"{self.scaler.__class__.__name__}_{model_name}"
            self.evaluation_[model_name] = {
                "mse": mean_squared_error(y_true, predictions),
                "r2": r2_score(y_true, predictions),
                "mae": mean_absolute_error(y_true, predictions),
            }
        self.evaluation_ = pd.DataFrame(self.evaluation_)


def plot_linear_regression_model(selector, X_train, X_test, y_train, y_test): # noqa
    coef = selector.fitted_models["LinearRegression"].named_steps["model"].coef_
    intercept = selector.fitted_models["LinearRegression"].named_steps["model"].intercept_
    linear_regression_equation = f"$y(x_1, x_2) = {intercept:0.3f} + {coef[0]:0.3f}  x_1 + {coef[1]:0.3f}  x_2 $"
    print(linear_regression_equation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16*0.7, 9*0.7))
    ax1.scatter(X_train["x1"], y_train, label="Train", color="gray", marker=".")
    ax1.scatter(X_test["x1"], y_test, label="True", color=TESCO_COLORS["blue"], marker="o")
    ax1.scatter(X_test["x1"], selector.predictions_["LinearRegression"], label="Predicted", color=TESCO_COLORS["red"], marker="o", s=100, alpha=0.5)
    x1_line = np.linspace(X_train["x1"].min(), X_train["x1"].max(), 100)
    x2_line = np.linspace(X_train["x2"].min(), X_train["x2"].max(), 100)
    y = intercept + coef[0] * x1_line + coef[1] * x2_line
    ax1.plot(x1_line, y, label=linear_regression_equation, color=TESCO_COLORS["red"], linestyle="--", linewidth=3)
    ax1.set_xlabel("$x_1$", fontsize=16)
    ax1.set_ylabel("$y$", fontsize=16)
    ax2.scatter(X_train["x2"], y_train, label="Train", color="gray", marker=".")
    ax2.scatter(X_test["x2"], y_test, label="True", color=TESCO_COLORS["blue"], marker="o")
    ax2.scatter(X_test["x2"], selector.predictions_["LinearRegression"], label="Predicted", color=TESCO_COLORS["red"], marker="o", s=100, alpha=0.5)
    ax2.plot(x2_line, y, label="$y(x_1, x_2)$", color=TESCO_COLORS["red"], linestyle="--", linewidth=3)
    ax2.set_xlabel("$x_2$", fontsize=18)
    ax2.set_ylabel("$y$", fontsize=18)
    fig.suptitle(f"Simplified model: {linear_regression_equation}", fontsize=18)
    ax2.legend(fontsize=14)
    plt.tight_layout()
    return fig


# %%
if __name__ == "__main__":
    # %%
    random_state = 1919
    df = load_preprocessed_data("masked_dataset")
    X = df[["x1", "x2"]]
    y = df["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    models = [
        RandomForestRegressor(random_state=random_state),
        GradientBoostingRegressor(random_state=random_state),
        LinearRegression(),
        Ridge(random_state=random_state),
        DummyRegressor(),
    ]
    decompositions = [
        PCA(n_components=1, random_state=random_state),
        None
    ]
    scalers = [
        None
    ]
    evaluations = []
    selectors = []
    for decomposition in decompositions:
        for scaler in scalers:
            model_selector = ModelSelector(models, "Regression", decomposition, scaler)
            model_selector.fit(X_train, y_train)
            model_selector.predict(X_test)
            model_selector.evaluate(y_test)
            evaluations.append(model_selector.evaluation_)
            selectors.append(model_selector)
    evaluations = pd.concat(evaluations, axis=1).T.sort_values(by="mse")
    evaluations
    # %%
    simplest_model = selectors[1]
    fig = plot_linear_regression_model(simplest_model, X_train, X_test, y_train, y_test)
