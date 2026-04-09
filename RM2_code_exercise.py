import random
from collections import Counter
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    roc_curve,
    auc,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Loading the data in the same way as before
reviews, sentiment_ratings, product_types, helpfulness_ratings = [], [], [], []

with open("Compiled_Reviews.txt") as f:
    for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        sentiment_ratings.append(fields[1])
        product_types.append(fields[2])
        helpfulness_ratings.append(fields[3])

print(f"Loaded {len(reviews)} reviews")

# 2. one-hot encoding
tokenised_reviews = [re.findall(r"\b\w+\b|[!?]+", review.lower()) for review in reviews]

tokens = []
for s in tokenised_reviews:
    tokens.extend(s)

counts = Counter(tokens)
so = sorted(counts.items(), key=lambda item: item[1], reverse=True)
so = list(zip(*so))[0]
type_list = so[:5000]

# Build one-hot matrix
word_to_idx = {word: j for j, word in enumerate(type_list)}
M = np.zeros((len(reviews), len(type_list)))
for i, tokenised_rev in enumerate(tokenised_reviews):
    rev_set = set(tokenised_rev)          # O(1) membership test
    for word in rev_set:
        if word in word_to_idx:
            M[i, word_to_idx[word]] = 1.0

print(f"Feature matrix shape: {M.shape}")

# 3. Train/dev/test split using sklearn package
M_train, M_remaining, sents_train, sents_remaining = train_test_split(
    M, sentiment_ratings, test_size=0.2, random_state=42
)
M_test, M_dev, sents_test, sents_dev = train_test_split(
    M_remaining, sents_remaining, test_size=0.5, random_state=42
)
print(f"Train: {M_train.shape}, Dev: {M_dev.shape}, Test: {M_test.shape}")

sents_test_binary = np.array([1 if s == "positive" else 0 for s in sents_test])

# 4.Model definition using Pytorch
class logistic_regression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# 5. Here，before training, I defined 2 helper functions to generate batches and evaluate the model
def make_dataloader(X, sents, batch_size):
    inputs_all = torch.from_numpy(X).float()
    labels_all = torch.from_numpy(
        np.array([int(l == "positive") for l in sents])
    ).float().unsqueeze(1)
    dataset = TensorDataset(inputs_all, labels_all)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def evaluate(model, X_test, y_test_binary):
    model.eval()
    with torch.no_grad():
        inputs_test = torch.from_numpy(X_test).float().to(device)
        outputs = model(inputs_test).squeeze().cpu().numpy()

    y_pred = (outputs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_binary, y_pred, average="macro"
    )
    acc = accuracy_score(y_test_binary, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    return acc, precision, recall, f1, outputs


# 6. Training and Evaluating Model 1: SGD, lr=0.01, lambda_l2=0.001
print("\nTraining Model 1 (SGD)...")
np.random.seed(42)

n_iters = 500
batch_size = 256
lambda_l2 = 0.001

model1 = logistic_regression(M_train.shape[1], 1).to(device)
criterion = nn.BCELoss()
optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
dataloader1 = make_dataloader(M_train, sents_train, batch_size)

loss_log_m1 = []
for epoch in range(n_iters):
    total_loss = 0
    model1.train()
    for inputs, labels in dataloader1:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        l2_reg = lambda_l2 * sum(p.pow(2.0).sum() for p in model1.parameters())
        loss += l2_reg
        total_loss += loss.item()
        loss.backward()
        optimizer1.step()
    loss_log_m1.append(total_loss / len(dataloader1))

    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1}/{n_iters}, Loss: {loss_log_m1[-1]:.4f}")

plt.figure()
plt.plot(range(1, n_iters + 1), loss_log_m1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Model 1: SGD, lr=0.01, lambda_l2={lambda_l2} (batch_size={batch_size})")
plt.savefig("loss_model1.png")
plt.close()
print("Saved loss_model1.png")

print("\n=== Model 1 Results ===")
acc_m1, prec_m1, rec_m1, f1_m1, M1_predictions = evaluate(model1, M_test, sents_test_binary)

# 7. Training and EvaluatingModel 2: Adam, lr=0.001, lambda_l2=0.0001
print("\nTraining Model 2 (Adam)...")

n_iters_2 = 500
batch_size_2 = 256
lambda_l2_2 = 0.0001

model2 = logistic_regression(M_train.shape[1], 1).to(device)
criterion2 = nn.BCELoss()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
dataloader2 = make_dataloader(M_train, sents_train, batch_size_2)

loss_log_m2 = []
for epoch in range(n_iters_2):
    total_loss = 0
    model2.train()
    for inputs, labels in dataloader2:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion2(outputs, labels)
        l2_reg = lambda_l2_2 * sum(p.pow(2.0).sum() for p in model2.parameters())
        loss += l2_reg
        total_loss += loss.item()
        loss.backward()
        optimizer2.step()
    loss_log_m2.append(total_loss / len(dataloader2))

    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1}/{n_iters_2}, Loss: {loss_log_m2[-1]:.4f}")

plt.figure()
plt.plot(range(1, n_iters_2 + 1), loss_log_m2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title(f"Model 2: Adam, lr=0.001, lambda_l2={lambda_l2_2} (batch_size={batch_size_2})")
plt.savefig("loss_model2.png")
plt.close()
print("Saved loss_model2.png")

print("\n=== Model 2 Results ===")
acc_m2, prec_m2, rec_m2, f1_m2, M2_predictions = evaluate(model2, M_test, sents_test_binary)

# 8. Printing out a comparison table
print(f"\n=== Comparison ===")
print(f"{'Metric':<12} {'Model 1':>10} {'Model 2':>10} {'Delta':>10}")
print(f"{'-'*44}")
for name, v1, v2 in [
    ("Accuracy",  acc_m1,  acc_m2),
    ("Precision", prec_m1, prec_m2),
    ("Recall",    rec_m1,  rec_m2),
    ("F1 Score",  f1_m1,   f1_m2),
]:
    print(f"{name:<12} {v1:>10.4f} {v2:>10.4f} {v2-v1:>+10.4f}")

# 9. Drawing ROC curves and calculating the AUC
fpr1, tpr1, _ = roc_curve(sents_test_binary, M1_predictions)
fpr2, tpr2, _ = roc_curve(sents_test_binary, M2_predictions)
auc_m1 = auc(fpr1, tpr1)
auc_m2 = auc(fpr2, tpr2)

plt.figure()
plt.plot(fpr1, tpr1, label=f"Model 1 (AUC={auc_m1:.4f})")
plt.plot(fpr2, tpr2, label=f"Model 2 (AUC={auc_m2:.4f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_comparison.png")
plt.close()
print("\nSaved roc_comparison.png")
print(f"AUC - Model 1: {auc_m1:.4f}, Model 2: {auc_m2:.4f}")

# 10. Bootstraping AUC difference to get a p value and confidence interval
def draw_bootstrap_sample(data):
    n = len(data[0])
    indices = random.choices(range(n), k=n)
    return tuple(arr[indices] for arr in data)


def bootstrap_auc_diff(y_test, M1_predictions, M2_predictions, num_samples=1000):
    diffs = []
    for _ in range(num_samples):
        y_bs, M1_bs, M2_bs = draw_bootstrap_sample((y_test, M1_predictions, M2_predictions))
        fpr1b, tpr1b, _ = roc_curve(y_bs, M1_bs)
        fpr2b, tpr2b, _ = roc_curve(y_bs, M2_bs)
        diffs.append(auc(fpr2b, tpr2b) - auc(fpr1b, tpr1b))

    diffs = np.array(diffs)
    # One-tailed p-value for H0: AUC_M2 <= AUC_M1 
    p_value = float(np.mean(diffs <= 0))
    diffs_sorted = np.sort(diffs)
    ci_low  = diffs_sorted[int(len(diffs_sorted) * 0.025)]
    ci_high = diffs_sorted[int(len(diffs_sorted) * 0.975)]
    return float(np.mean(diffs)), ci_low, ci_high, p_value


print("\nRunning bootstrap (1000 samples)...")
mean_diff, ci_low, ci_high, p_value = bootstrap_auc_diff(
    sents_test_binary, M1_predictions, M2_predictions, num_samples=1000
)
print(f"AUC difference (M2 - M1): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"One-tailed p-value (H1: AUC_M2 > AUC_M1): {p_value:.4f}")
if p_value < 0.05:
    print("=> Model 2 AUC is significantly higher than Model 1 (p < 0.05)")
else:
    print("=> No significant difference in AUC at p < 0.05")
