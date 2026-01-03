import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RandomForestModel:

    def __init__(self, x_train_path, y_train_path, model_output_path, metrics_path):
        self.x_train_path = x_train_path
        self.y_train_path = y_train_path
        self.model_output_path = model_output_path
        self.metrics_path = metrics_path
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    def load_data(self):
        X_train = pd.read_csv(self.x_train_path)
        y_train = pd.read_csv(self.y_train_path)
        return X_train, y_train.squeeze()

    def train(self):
        X_train, y_train = self.load_data()
        self.model.fit(X_train, y_train)

    def evaluate(self, X_train, y_train):
        preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, preds)
        return acc

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(self.model, self.model_output_path)

    def save_metrics(self, accuracy):
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
        with open(self.metrics_path, "w") as f:
            f.write(f"accuracy: {accuracy}")

    def run(self):
        X_train, y_train = self.load_data()
        self.model.fit(X_train, y_train)
        accuracy = self.evaluate(X_train, y_train)
        self.save_model()
        self.save_metrics(accuracy)
        print(f"Random Forest training complete | Accuracy: {accuracy}")
