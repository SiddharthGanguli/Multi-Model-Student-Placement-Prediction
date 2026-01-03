from decision_tree import DecisionTreeModel

def main():
    model = DecisionTreeModel(
        x_train_path="Data/Split/X_train.csv",
        y_train_path="Data/Split/y_train.csv",
        model_output_path="models/decision_tree.pkl",
        metrics_path="metrics/decision_tree.txt"
    )
    model.run()


if __name__ == "__main__":
    main()
