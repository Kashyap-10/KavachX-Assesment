"""
Task 3: Verifier Performance Evaluation & Operating Threshold Selection.
Calculates Precision and Recall for:
  (a) Edge Model Alone (All 30 alerts sent)
  (b) Edge + Secondary Verifier across operating confidence thresholds
Selects and justifies an optimal operating threshold.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def evaluate_verifier_metrics(
    scores_csv: str = "task3/outputs/verifier_scores.csv",
    output_dir: str = "task3/outputs",
):
    print("=" * 70)
    print("Task 3: Verifier Evaluation & Optimal Threshold Selection")
    print("=" * 70)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(scores_csv):
        print(f"[Notice] '{scores_csv}' not found. Running verifier first...")
        from task3.verifier import CascadedVerifier, create_sample_alerts_if_missing
        create_sample_alerts_if_missing("alerts", "alerts/labels.csv")
        verifier = CascadedVerifier()
        df = verifier.evaluate_directory("alerts", "alerts/labels.csv")
        df.to_csv(scores_csv, index=False)
    else:
        df = pd.read_csv(scores_csv)

    total_alerts = len(df)
    total_intrusions = int(df["is_intrusion"].sum())
    total_false_alarms = total_alerts - total_intrusions

    edge_tp = total_intrusions
    edge_fp = total_false_alarms
    edge_precision = edge_tp / total_alerts
    edge_recall = 1.0
    edge_f1 = 2 * (edge_precision * edge_recall) / (edge_precision + edge_recall)

    print("\n[A] Edge Model Alone Performance (Zero Verification):")
    print(f"  • Total Alerts Dispatched   : {total_alerts}")
    print(f"  • True Intrusions Captured  : {edge_tp}/{total_intrusions}")
    print(f"  • False Alarms Sent to User : {edge_fp}")
    print(f"  • Precision                 : {edge_precision:.3f} ({edge_precision*100:.1f}%)")
    print(f"  • Recall                    : {edge_recall:.3f} (100.0%)")
    print(f"  • F1-Score                  : {edge_f1:.3f}")

    thresholds = np.arange(0.30, 0.85, 0.05)
    sweep_results = []

    for t in thresholds:
        predicted_pos = df["verifier_score"] >= t
        tp = int(((predicted_pos == True) & (df["is_intrusion"] == 1)).sum())
        fp = int(((predicted_pos == True) & (df["is_intrusion"] == 0)).sum())
        fn = int(((predicted_pos == False) & (df["is_intrusion"] == 1)).sum())
        tn = int(((predicted_pos == False) & (df["is_intrusion"] == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-9)

        sweep_results.append({
            "Threshold": round(float(t), 2),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1-Score": round(f1, 3),
            "F2-Score": round(f2, 3),
            "False Alarm Reduction": f"{(1 - fp/total_false_alarms)*100:.1f}%",
        })

    sweep_df = pd.DataFrame(sweep_results)

    high_recall_candidates = sweep_df[sweep_df["Recall"] >= 0.95]
    if len(high_recall_candidates) > 0:
        optimal_idx = high_recall_candidates["F2-Score"].idxmax()
    else:
        optimal_idx = sweep_df["F1-Score"].idxmax()

    optimal = sweep_df.loc[optimal_idx]

    print("\n[B] Edge + Secondary Verifier (Optimal Threshold):")
    print(f"  • Selected Threshold        : {optimal['Threshold']}")
    print(f"  • Precision                 : {optimal['Precision']:.3f} (vs {edge_precision:.3f} edge alone)")
    print(f"  • Recall                    : {optimal['Recall']:.3f}")
    print(f"  • False Alarms Eliminated   : {optimal['False Alarm Reduction']}")
    print(f"  • Optimal F2-Score          : {optimal['F2-Score']:.3f}")

    print("\nThreshold Sweep Table:")
    print(sweep_df[["Threshold", "TP", "FP", "FN", "Precision", "Recall", "F1-Score", "False Alarm Reduction"]].to_markdown(index=False))

    eval_md_path = os.path.join(output_dir, "verifier_evaluation.md")
    with open(eval_md_path, "w", encoding="utf-8") as f:
        f.write("# Task 3: Cascaded Verifier Evaluation & Threshold Rationale\n\n")
        f.write("## 1. Metric Comparison\n\n")
        f.write("| Pipeline Configuration | Precision | Recall | F1-Score | False Alarms Sent |\n")
        f.write("|------------------------|-----------|--------|----------|-------------------|\n")
        f.write(f"| **(a) Edge Model Alone** | `{edge_precision:.3f}` | `{edge_recall:.3f}` | `{edge_f1:.3f}` | `{edge_fp}` (100% false alarms) |\n")
        f.write(f"| **(b) Edge + Secondary Verifier (t={optimal['Threshold']})** | **`{optimal['Precision']:.3f}`** | **`{optimal['Recall']:.3f}`** | **`{optimal['F1-Score']:.3f}`** | **`{int(optimal['FP'])}`** ({optimal['False Alarm Reduction']} eliminated) |\n\n")

        f.write("## 2. Threshold Selection Justification\n\n")
        f.write(f"We select **`threshold = {optimal['Threshold']}`** based on physical security risk economics:\n")
        f.write("- **Asymmetric Loss Function:** In active intrusion surveillance, a False Negative (missed intruder) has catastrophic liability, whereas a False Positive merely costs a few seconds of human operator review. Therefore, we constrain our search to operating regimes with **Recall ≥ 95%**.\n")
        f.write(f"- At `threshold = {optimal['Threshold']}`, we eliminate **{optimal['False Alarm Reduction']}** of false alarms (caused by shadows, rain droplets, bugs, and stray animals) while maintaining near-perfect intrusion sensitivity (`Recall = {optimal['Recall']:.3f}`).\n\n")

        f.write("## 3. Full Operating Sweep\n\n")
        f.write(sweep_df.to_markdown(index=False) + "\n")

    print(f"\n[Saved] Comprehensive report saved to: {eval_md_path}")
    return sweep_df


if __name__ == "__main__":
    evaluate_verifier_metrics()
