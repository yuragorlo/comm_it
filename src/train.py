import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from math import sqrt
import lightgbm as lgb


class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.selected_columns = None

    def fit(self, X, y):
        correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        self.selected_columns = np.where(correlations >= self.threshold)[0]
        return self

    def transform(self, X):
        return X[:, self.selected_columns]


def get_pipeline():
    normalize_columns = [7, 8]
    standardize_columns = list(set(range(53)) - set(normalize_columns))

    preprocessor = ColumnTransformer(
        transformers=[
            ('standardize', StandardScaler(), standardize_columns),
            ('normalize', MinMaxScaler(), normalize_columns)
        ],
        remainder='passthrough'
    )

    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    pca = PCA(n_components=0.95)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('polynomial', poly_features),
        ('correlation_filter', CorrelationFilter(threshold=0.003)),
        ('pca', pca),
        ('regressor', lgb.LGBMRegressor(
            random_state=42,
            verbosity=-1,
            bagging_fraction=0.8739463660626885,
            colsample_bytree=0.8721230154351118,
            learning_rate=0.16009985039390862,
            max_depth=11,
            min_child_samples=28,
            n_estimators=338,
            num_leaves=23
        ))
    ])

    return pipeline


def train_model():
    train_df = pd.read_csv("../dataset/train.csv")
    X = train_df.drop(columns=['target'])
    y = train_df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)

    rmse_train = sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_val = sqrt(mean_squared_error(y_val, y_val_pred))

    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Validation RMSE: {rmse_val:.4f}")

    return pipeline


if __name__ == "__main__":
    train_model()