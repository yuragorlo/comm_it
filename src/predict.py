import pandas as pd
from train import get_pipeline


def generate_predictions():
    train_df = pd.read_csv("../dataset/train.csv")
    test_df = pd.read_csv("../dataset/hidden_test.csv")
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df

    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    results_df = pd.DataFrame({'target': y_pred})
    results_df.to_csv("../dataset/predictions.csv", index=False)
    print("Predictions saved to ../dataset/predictions.csv")


if __name__ == "__main__":
    generate_predictions()