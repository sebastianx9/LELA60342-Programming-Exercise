# Sentiment Classification – LELA60342 Coding Exercise

**Student ID:** 11479116

## Overview

This project trains two logistic regression classifiers to predict sentiment (positive/negative) from Amazon product reviews, using a bag-of-words representation built from the 5000 most frequent tokens.

## Models

**Model 1** uses SGD with a learning rate of 0.01 and L2 regularisation (λ = 0.001), trained for 500 epochs with a batch size of 256.

**Model 2** replaces the optimiser with Adam (lr = 0.001) and reduces regularisation (λ = 0.0001), keeping the same architecture and number of epochs. The switch to Adam was motivated by its adaptive learning rates, which tend to converge more stably on sparse input features like one-hot encodings.

## Results

| Metric    | Model 1 | Model 2 |
|-----------|---------|---------|
| Accuracy  | 0.8484  | 0.8569  |
| Precision | 0.8467  | 0.8541  |
| Recall    | 0.8424  | 0.8531  |
| F1 Score  | 0.8442  | 0.8536  |
| AUC       | 0.9196  | 0.9295  |

Model 2 outperforms Model 1 across all metrics.

## Significance Testing

A bootstrap test (1000 samples) was used to assess whether the AUC improvement is statistically significant. The mean AUC difference (M2 − M1) was 0.0100, with a 95% confidence interval of [0.0063, 0.0136]. The one-tailed p-value was < 0.001, indicating that Model 2's AUC is significantly higher than Model 1's.

## Files

- `RM2_code_exercise.py` – full training and evaluation pipeline
- `run.sh` – Slurm submission script
- `slurm-13383344.out` – output from the CSF run
- `loss_model1.png` – training loss curve for Model 1
- `loss_model2.png` – training loss curve for Model 2
- `roc_comparison.png` – ROC curve comparison
