class LagBaselineRegressor:
    model_source = "lag_1_baseline"

    def predict(self, X):
        return X["lag_1"].to_numpy()
