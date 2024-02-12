# %%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from tesco.data.preprocessing import load_preprocessed_data

df = load_preprocessed_data("masked_dataset")

df
# %%
linear_model_pca = LinearRegression()
linear_model = LinearRegression()
random_forest = RandomForestRegressor()
pca = PCA(n_components=1)


X = df[["x1", "x2"]]
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

linear_model.fit(X_train, y_train)
linear_model_pca.fit(X_train_pca, y_train)
random_forest.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
y_pred_linear_pca = linear_model_pca.predict(X_test_pca)
y_pred_rf = random_forest.predict(X_test)

mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_linear_pca = mean_squared_error(y_test, y_pred_linear_pca)
mse_rf = mean_squared_error(y_test, y_pred_rf)


# %%

mse_linear, mse_linear_pca, mse_rf

# %%
# plt.plot(X_train_pca, y_train, "ro", label="Train")
plt.plot(X_test_pca, y_test, "bo", label="Real Data")
plt.plot(X_test_pca, y_pred_linear, "go", label="Linear Regression")
plt.plot(X_test_pca, y_pred_rf, "yo", label="Random Forest")
plt.legend()
# %%
pca.explained_variance_ratio_
# %%
linear_eq = (
    f"y(x1, x2) = {linear_model.intercept_:0.3f} + {linear_model.coef_[0]:0.3f}x1 + {linear_model.coef_[1]:0.3f}x2"
)
linear_eq
# %%
