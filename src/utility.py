"""Utility functions for machine learning models."""
# pylint: disable=invalid-name
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import roc_curve, auc

def train_timer(model: object, X: pd.DataFrame, Y: pd.Series) -> float:
    """Times how long it takes to train a model in milliseconds."""
    start_time = time.time()
    model.fit(X, Y)
    end_time = time.time()
    train_time = (end_time - start_time) * 1000
    print(f'Model training time: {train_time:.2f} ms')
    return train_time

def fit_eval(model, X, Y):
    """Fits a model and evaluates its performance metrics.
    
    Returns predictions, prediction probabilities, and metrics series."""
    Yhat = model.predict(X)
    Yhat_prob = model.predict_proba(X)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(Y, Yhat),
        'Precision': precision_score(Y, Yhat),
        'Recall': recall_score(Y, Yhat),
        'Specificity': recall_score(Y, Yhat, pos_label=0)
    }
    return Yhat, Yhat_prob, pd.Series(metrics)

def print_metrics(metrics_series):
    """Prints formatted model evaluation metrics."""
    print('------------- MODEL METRICS --------------')
    for metric, value in metrics_series.items():
        print(f'{metric}: {value:.2f}')

def plot_evaluation_graphs(Y, Yhat, Yhat_probs):
    """Plots confusion matrix and ROC curve for model evaluation."""
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Confusion Matrix
    cmd.from_predictions(
        Y,
        Yhat,
        display_labels=['Perished', 'Survived'],
        cmap='Blues',
        ax=ax1
    )
    ax1.set_title('Confusion Matrix')

    # ROC Curve
    fpr, tpr, _ = roc_curve(Y, Yhat_probs)
    roc_auc = auc(fpr, tpr)
    ax2.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve')
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

def train_set_evaluation(model, X_train, Y_train, X_test, Y_test):
    """Evaluates model performance with increasing training set sizes.
    
    Returns DataFrame with training set sizes and corresponding accuracies."""
    training_set = []
    train_acc = []
    test_acc = []
    for i in range(10, len(X_train) + 1, 10):
        model.fit(X_train.iloc[:i], Y_train.iloc[:i])
        Yhat_train = model.predict(X_train.iloc[:i])
        Yhat_test = model.predict(X_test)

        training_set.append(i)
        train_acc.append(accuracy_score(Y_train.iloc[:i], Yhat_train))
        test_acc.append(accuracy_score(Y_test, Yhat_test))

    return pd.DataFrame(
        {
            'Training Set Size': training_set, 
            'Train Accuracy': train_acc, 
            'Test Accuracy': test_acc
        }
    )

def plot_accuracy(results_df):
    """Plots training and test accuracy vs training set size."""
    plt.figure(figsize=(8, 4))
    sns.lineplot(
        x=results_df['Training Set Size'],
        y=results_df['Train Accuracy'],
        label='Train Accuracy'
    )
    sns.lineplot(
        x=results_df['Training Set Size'],
        y=results_df['Test Accuracy'],
        label='Test Accuracy'
    )
    plt.title('Train and Test Accuracy vs Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
