"""Develop a model to predict the normalised sales of a store for a given location."""
# %%

import pandas as pd 

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tesco.data.preprocessing import load_preprocessed_data

# %%
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data for model training."""
    # Drop columns that are not useful for model training
    df = df.drop(
        columns=[
            "county",
            "location_id",
            "is_train",
        ]
    )

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["new_store", "store_type"])

    return df

# %%