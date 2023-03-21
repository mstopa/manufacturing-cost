import argparse

from failure_predictor.classifiers import SklearnClassifier
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model.joblib")
    args = parser.parse_args()
    
    classifier = SklearnClassifier(args.model_path)
    df = pd.read_csv("data/processed_data.csv")

    sample = df.sample(1)
    label, features = sample["Response"].values[0], sample.drop(["Id", "Response", "subset"], axis=1).values[0]
    
    print(f"Predicted: {classifier.predict(features)} for label: {label}")