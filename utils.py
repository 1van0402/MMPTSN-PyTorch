import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, cohen_kappa_score,
                             recall_score, f1_score, precision_score)

def calculate_metrics(y_true, y_pred):
    mat = confusion_matrix(y_true, y_pred)

    # Kappa、OA
    Kappa = cohen_kappa_score(y_true, y_pred)
    OA    = accuracy_score(y_true, y_pred) * 100

    CA = np.diag(mat) / np.sum(mat, axis=0) * 100
    recall_per_class = recall_score(y_true, y_pred, average=None) * 100
    f1_per_class = f1_score(y_true, y_pred, average=None) * 100
    macro_recall = recall_score(y_true, y_pred, average='macro') * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro') * 100

    acc = [OA, Kappa] + CA.tolist() + recall_per_class.tolist() + f1_per_class.tolist() + [macro_recall, macro_f1]

    return acc
def save_classification_image():
    return None