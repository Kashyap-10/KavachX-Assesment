# Task 3: Cascaded Verifier Evaluation & Threshold Rationale

## 1. Metric Comparison

| Pipeline Configuration | Precision | Recall | F1-Score | False Alarms Sent |
|------------------------|-----------|--------|----------|-------------------|
| **(a) Edge Model Alone** | `0.433` | `1.000` | `0.605` | `17` (100% false alarms) |
| **(b) Edge + Secondary Verifier (t=0.6)** | **`1.000`** | **`1.000`** | **`1.000`** | **`0`** (100.0% eliminated) |

## 2. Threshold Selection Justification

We select **`threshold = 0.6`** based on physical security risk economics:
- **Asymmetric Loss Function:** In active intrusion surveillance, a False Negative (missed intruder) has catastrophic liability, whereas a False Positive merely costs a few seconds of human operator review. Therefore, we constrain our search to operating regimes with **Recall ≥ 95%**.
- At `threshold = 0.6`, we eliminate **100.0%** of false alarms (caused by shadows, rain droplets, bugs, and stray animals) while maintaining near-perfect intrusion sensitivity (`Recall = 1.000`).

## 3. Full Operating Sweep

|   Threshold |   TP |   FP |   FN |   TN |   Precision |   Recall |   F1-Score |   F2-Score | False Alarm Reduction   |
|------------:|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|-----------:|:------------------------|
|        0.3  |   13 |   17 |    0 |    0 |       0.433 |    1     |      0.605 |      0.793 | 0.0%                    |
|        0.35 |   13 |   17 |    0 |    0 |       0.433 |    1     |      0.605 |      0.793 | 0.0%                    |
|        0.4  |   13 |   17 |    0 |    0 |       0.433 |    1     |      0.605 |      0.793 | 0.0%                    |
|        0.45 |   13 |   17 |    0 |    0 |       0.433 |    1     |      0.605 |      0.793 | 0.0%                    |
|        0.5  |   13 |   14 |    0 |    3 |       0.481 |    1     |      0.65  |      0.823 | 17.6%                   |
|        0.55 |   13 |   10 |    0 |    7 |       0.565 |    1     |      0.722 |      0.867 | 41.2%                   |
|        0.6  |   13 |    0 |    0 |   17 |       1     |    1     |      1     |      1     | 100.0%                  |
|        0.65 |   13 |    0 |    0 |   17 |       1     |    1     |      1     |      1     | 100.0%                  |
|        0.7  |   13 |    0 |    0 |   17 |       1     |    1     |      1     |      1     | 100.0%                  |
|        0.75 |    6 |    0 |    7 |   17 |       1     |    0.462 |      0.632 |      0.517 | 100.0%                  |
|        0.8  |    0 |    0 |   13 |   17 |       0     |    0     |      0     |      0     | 100.0%                  |
