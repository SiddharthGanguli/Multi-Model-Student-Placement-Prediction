import pandas as pd
from sklearn.model_selection import train_test_split
import os

class Split:

    def __init__(self, input_path, output_dir, test_size=0.2, random_state=42):
        self.input_path = input_path
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.input_path)

    def split(self):
        X = self.df.drop("Placement_Status", axis=1)
        y = self.df["Placement_Status"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def save_data(self, X_train, X_test, y_train, y_test):
        os.makedirs(self.output_dir, exist_ok=True)

        X_train.to_csv(f"{self.output_dir}/X_train.csv", index=False)
        X_test.to_csv(f"{self.output_dir}/X_test.csv", index=False)
        y_train.to_csv(f"{self.output_dir}/y_train.csv", index=False)
        y_test.to_csv(f"{self.output_dir}/y_test.csv", index=False)

    def run(self):
        self.load_data()
        X_train, X_test, y_train, y_test = self.split()
        self.save_data(X_train, X_test, y_train, y_test)
