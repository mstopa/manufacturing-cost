import argparse

from classifiers import SklearnClassifier
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.joblib")
    args = parser.parse_args()
    
    classifier = SklearnClassifier(args.model_path)
    df = pd.read_csv("data/processed_data.csv")

    feature_names = df.drop(["Id", "Response", "subset"], axis=1).columns
    with open("feats.txt", "w") as f:
        f.write("[")
        for feat in feature_names:
            f.write(f"'{feat}', ")
        f.write("]")

    sample = df.sample(1)
    label, features = sample["Response"].values[0], sample.drop(["Id", "Response", "subset"], axis=1).values[0]
    
    print(f"Predicted: {classifier.predict(features)} for label: {label}")