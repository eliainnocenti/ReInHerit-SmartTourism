import os

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def normalize_scores(scores, inverse=False):
    min_score = min(scores)
    max_score = max(scores)
    if inverse:
        return [1 - ((s - min_score) / (max_score - min_score)) for s in scores]
    else:
        return [(s - min_score) / (max_score - min_score) for s in scores]


def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    return accuracy, precision, recall, f1


# Load the data
data_path = os.path.join('../', 'runs', 'scores.csv')
data = pd.read_csv(data_path)

# Normalize the scores
od_scores = normalize_scores(data[['OD_Score1', 'OD_Score2', 'OD_Score3']].values.flatten())
ir_scores = normalize_scores(data[['IR_Score1', 'IR_Score2', 'IR_Score3']].values.flatten(), inverse=True)

# Reorder the scores
data[['OD_NormScore1', 'OD_NormScore2', 'OD_NormScore3']] = np.array(od_scores).reshape(-1, 3)
data[['IR_NormScore1', 'IR_NormScore2', 'IR_NormScore3']] = np.array(ir_scores).reshape(-1, 3)

# Calculate the metrics for Object Detection
od_correct = (data['True_Monument'] == data['OD_Top1'])
od_accuracy, od_precision, od_recall, od_f1 = calculate_metrics(data['True_Monument'], data['OD_Top1'])

# Calculate the metrics for Image Retrieval
ir_correct = (data['True_Monument'] == data['IR_Top1'])
ir_accuracy, ir_precision, ir_recall, ir_f1 = calculate_metrics(data['True_Monument'], data['IR_Top1'])

# Print the results
print("Object Detection Results:")
print(f"Accuracy: {od_accuracy:.4f}")
print(f"Precision: {od_precision:.4f}")
print(f"Recall: {od_recall:.4f}")
print(f"F1-Score: {od_f1:.4f}")

print("\nImage Retrieval Results:")
print(f"Accuracy: {ir_accuracy:.4f}")
print(f"Precision: {ir_precision:.4f}")
print(f"Recall: {ir_recall:.4f}")
print(f"F1-Score: {ir_f1:.4f}")

# Analysis of cases where one technique outperformed the other
od_better = data[od_correct & ~ir_correct]
ir_better = data[ir_correct & ~od_correct]

print(f"\nCases where Object Detection outperformed Image Retrieval: {len(od_better)}")
print(f"Cases where Image Retrieval outperformed Object Detection: {len(ir_better)}")

# Calculate the precision@K for K=1,2,3
for k in [1, 2, 3]:
    od_precision_at_k = (
                (data[['OD_Top1', 'OD_Top2', 'OD_Top3']].iloc[:, :k] == data['True_Monument'].values[:, None]).sum(
                    axis=1) > 0).mean()
    ir_precision_at_k = (
                (data[['IR_Top1', 'IR_Top2', 'IR_Top3']].iloc[:, :k] == data['True_Monument'].values[:, None]).sum(
                    axis=1) > 0).mean()

    print(f"\nPrecision@{k}:")
    print(f"Object Detection: {od_precision_at_k:.4f}")
    print(f"Image Retrieval: {ir_precision_at_k:.4f}")