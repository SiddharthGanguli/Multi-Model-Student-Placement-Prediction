from split import Split

def main():
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
