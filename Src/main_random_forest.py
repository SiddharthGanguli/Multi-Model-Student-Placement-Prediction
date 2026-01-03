from random_forest import RandomForestModel

def main():
    model = RandomForestModel(
        x_train_path="Data/Split/X_train.csv",
        y_train_path="Data/Split/y_train.csv",
        model_output_path="models/random_forest.pkl",
        metrics_path="metrics/random_forest.txt"
    )
    model.run()

if __name__ == "__main__":
    main()
