import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import joblib

class Preprocessing:

    def __init__(self, input_path, output_path, encoder_path, is_train=True):
        self.input_path = input_path
        self.output_path = output_path
        self.encoder_path = encoder_path
        self.is_train = is_train
        self.df = None
        self.encoders = {}


    def load_data(self):
        self.df = pd.read_csv(self.input_path)
        return self.df

    def clear_nullvalues(self):
        if self.df.isnull().sum().sum() > 0:
            num_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
            self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())

            cat_cols = self.df.select_dtypes(include=["object"]).columns
            for col in cat_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            print("Missing values handled successfully")
        else:
            print("No missing values found")

        return self.df

    def drop_cols(self):
        if "Student_ID" in self.df.columns:
            self.df = self.df.drop(columns=["Student_ID"])
        return self.df
    
    def encode_features(self):
        cat_cols = self.df.select_dtypes(include=["object"]).columns
        cat_cols = [col for col in cat_cols if col != "Placement_Status"]

        if self.is_train:
            for col in cat_cols:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
            self.save_encoders()
        else:
            self.encoders = joblib.load(f"{self.encoder_path}/feature_encoders.pkl")
            for col in cat_cols:
                self.df[col] = self.encoders[col].transform(self.df[col])

        return self.df


    def encode_target(self):
        if self.is_train:
            target_encoder = LabelEncoder()
            self.df["Placement_Status"] = target_encoder.fit_transform(
                self.df["Placement_Status"]
            )
            joblib.dump(target_encoder, f"{self.encoder_path}/target_encoder.pkl")
        else:
            target_encoder = joblib.load(f"{self.encoder_path}/target_encoder.pkl")
            self.df["Placement_Status"] = target_encoder.transform(
                self.df["Placement_Status"]
            )

    def save_encoders(self):
        os.makedirs(self.encoder_path, exist_ok=True)
        joblib.dump(self.encoders, f"{self.encoder_path}/feature_encoders.pkl")

    def save_data(self):
        output_dir = os.path.dirname(self.output_path)
        os.makedirs(output_dir, exist_ok=True)

        self.df.to_csv(self.output_path, index=False)
        print(f"Preprocessed data saved at: {self.output_path}")

    def run_preprocessing(self):
        self.load_data()
        self.clear_nullvalues()
        self.drop_cols()
        self.encode_features()
        self.encode_target()
        self.save_encoders()
        self.save_data()

        return self.df
