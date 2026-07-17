from src.preprocessing import load_and_preprocess
from src.train_models import train_models
from src.evaluate_models import evaluate_models
from src.visualize_results import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance
)

data_path = "data/diabetes.csv"

X_train, X_test, y_train, y_test, feature_names = load_and_preprocess(data_path)

models = train_models(X_train, y_train)

results = evaluate_models(models, X_test, y_test)

for model_name, metrics in results.items():

    print("\n===========================")
    print(model_name)
    print("===========================")

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1 Score:", metrics["f1_score"])

    plot_confusion_matrix(metrics["confusion_matrix"], model_name)

plot_model_comparison(results)

plot_feature_importance(models["Random Forest"], feature_names)
