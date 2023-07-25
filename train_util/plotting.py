import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from typing import List


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    """Plot the confusion matrix using a heatmap.

   :param y_true: True labels of the test set.
   :param y_pred: Predicted labels of the test set.
   :param class_names: List of class names.
   """

    fig, ax = plt.subplots(figsize=(22, 20))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names,
                                            normalize='true', xticks_rotation="vertical",
                                            ax=ax, colorbar=False)
    plt.title('Confusion matrix')
    plt.show(block=False)


def plot_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> None:
    """Plot a heatmap visualization of a classification report.

    :param y_true: True labels of the test set .
    :param y_pred: Predicted labels of the test set.
    :param class_names: A list of class names corresponding to the labels.
    """

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df = df.drop(['support'], axis=1)
    plt.figure(figsize=(18, 15))
    sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Classification Report Heatmap')
    plt.show(block=False)
