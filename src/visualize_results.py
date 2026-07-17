import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, model_name):

    plt.figure()

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.show()


def plot_model_comparison(results):

    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in model_names]

    plt.figure()

    plt.bar(model_names, accuracies)

    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)

    plt.show()


def plot_feature_importance(model, feature_names):

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        plt.figure()

        plt.barh(feature_names, importance)

        plt.title("Feature Importance (Random Forest)")

        plt.show()
