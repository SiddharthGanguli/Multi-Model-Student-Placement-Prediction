from preprocessing import Preprocessing
from split import Split

def main():


    train_pre = Preprocessing(
        input_path="Data/Raw_data/train.csv",
        output_path="Data/Processed/processed_train.csv",
        encoder_path="artifacts/encoders",
        is_train=True
    )
    train_pre.run_preprocessing()
    print("Train data preprocessing complete")


    test_pre = Preprocessing(
        input_path="Data/Raw_data/test.csv",
        output_path="Data/Processed/processed_test.csv",
        encoder_path="artifacts/encoders",
        is_train=False
    )
    test_pre.run_preprocessing()
    print("Test data preprocessing complete")

    splitter = Split(
        input_path="Data/Processed/processed_train.csv",
        output_dir="Data/Split/",
        test_size=0.2,
        random_state=42
    )

    splitter.run()
    print("Train-test split completed successfully")

    
if __name__ == "__main__":
    main()
