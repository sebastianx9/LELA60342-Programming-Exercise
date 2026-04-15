# LELA60342 Programming Exercise

**Student ID:** 11479116

## Overview

In model 1, modifications are made to train a sentiment classifier of Amazon reviews from the CL1 coursework with pytorch implementations including model specification, batch generation, the backwards step, loss function and optimisation. 

In a attempt to imporve the performance of model 1, the second model is created through a few adjustments in terms of hyperparameters. The features are one-hot endcoded bag-of-words vectors built from the 5000 most frequent tokens in the dataset. Both models are based on logistic regeression to predict the sentiment (positive or negative) of Amazon product reviews, and both are implemented in PyTorch and run on a GPU via CUDA. 

As for evaluation, besides accuracy and F1-scores, area under the ROC curves were calculated for both models to examine their discriminative power across all thresholds instead of a single cutoff at 50%. Such examination allows insight into how well the models perform with thresholds set catering to a particular need of the task. 

To test the significance of the models' difference in AUC, bootstrap sampling is used on the test set. With corresponding predictions from both models, the AUC difference and its variance are calculated. 

In the results, with the null hypothesis that Model 2’s AUC is no greater than Model 1’s AUC (H0:Diff=AUC_M2-AUC_M1<=0), the p‑value was found to be smaller than 0.05, which means that Model 2 yields a significantly greater AUC than Model 1. Hence, there is enough evidence to suggest that the adjustment made led to a significant improvement in the model's discriminative power.

---

## Data

The dataset has 36,548 Amazon reviews labelled as positive or negative. Reviews are tokenised with a regex that captures word tokens and punctuation like `!` and `?`. A binary vector of length 5000 is then constructed for each review, with a "1" wherever a token from the vocabulary appears.

The data is split as follows:
- Training: 80% (29,238 reviews)
- Development: 10% (3,655 reviews)
- Test: 10% (3,655 reviews)

---

## Models

Both models have the same architecture: a single linear layer followed by sigmoid function, and are trained with "BCELoss"(Binary corss entropy loss). L2 regularisation is added manually as a penalty on the weight norms during each training step.

### Model 1

- Optimiser: SGD, lr = 0.01
- L2 λ: 0.001
- Batch size: 256, Epochs: 500

SGD with a fixed learning rate serves as the baseline. The stronger L2 penalty is used to keep the weights small on the high-dimensional input.

### Model 2

- Optimiser: Adam, lr = 0.001
- L2 λ: 0.0001
- Batch size: 256, Epochs: 500

The main change is switching to the Adam optimizer, which adjusts the learning rate per parameter during training. One-hot features are very sparse (most values are zero), and Adam tends to handle sparse gradients better than plain SGD. The L2 penalty is lowered slightly since Adam is already more conservative in its updates.

---

## Significance Testing

A bootstrap test with 1000 resamples was run to check whether the AUC difference is statistically significant. In each resample, both models were evaluated on a random sample (with replacement) of the test set, and the AUC difference was recorded. With 1000 data points of AUC difference, we calculated the mean AUC difference (M2 − M1), the confidence interval and the one-tailed p value under the null hypothesis that Model 2’s AUC is no greater than Model 1’s AUC (H0:Diff=AUCM2-AUCM1<=0).


## Results 

### Model results

| Metric    | Model 1 | Model 2 | Delta   |
|-----------|---------|---------|---------|
| Accuracy  | 0.8484  | 0.8569  | +0.0085 |
| Precision | 0.8467  | 0.8541  | +0.0074 |
| Recall    | 0.8424  | 0.8531  | +0.0107 |
| F1 Score  | 0.8442  | 0.8536  | +0.0094 |
| AUC       | 0.9196  | 0.9295  | +0.0099 |

Model 2 is better on all metrics. The recall improvement (+0.0107) is the largest, which makes sense since Adam's per-parameter updates help the model pick up signal from less frequent tokens that SGD might underweight. Both models have AUC above 0.91, so the bag-of-words features on their own already carry a lot of sentiment information. The gains from switching optimiser are real but modest, which is expected given that the architecture and features are unchanged.

### Significance testing result

results:
- Mean AUC difference (M2 − M1): 0.0100
- 95% CI: [0.0063, 0.0136]
- One-tailed p-value: < 0.001

As we can see in the results, the confidence interval sits entirely above zero and the p-value is well below 0.05. As the proportion of 1,000 bootstrap resamples of the test set in which AUC_M2-AUC_M1<=0, the low p-value leads to rejection of the null hypothesis, indicating that Model 2 has significantly higher discriminative power than Model 1.

All in all, in this experiment, we successfully improved performance of a sentiment classifier and used boostrapping to test the significance of such improvement.
---

## Files

- `RM2_code_exercise.py` – training and evaluation code
- `run.sh` – Slurm script
- `slurm-13383344.out` – full output from the CSF run
- `loss_model1.png` – Model 1 training loss curve
- `loss_model2.png` – Model 2 training loss curve
- `roc_comparison.png` – ROC curves for both models
