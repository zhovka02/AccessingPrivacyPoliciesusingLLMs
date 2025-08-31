"""
Evaluation and comparison utilities for LLM classification results.

This script loads two combined JSON result files (per-model outputs), computes
per-label precision/recall/F1 metrics, prints formatted reports, and provides a
delta comparison between the two runs.

Outputs are streamed both to console and to a log file using a Tee class.
Optionally, metrics can also be written to JSON files.
"""

import json
import argparse
import sys
from sklearn.metrics import classification_report
from promt_templates import FULL_LABEL_NAMES


class Tee:
    """
    Simple tee-like stream duplicator.

    Writes all stdout output to multiple underlying streams simultaneously,
    typically sys.stdout and a log file.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def evaluate(combined, output_path=None):
    """
    Compute classification metrics for each model.

    Parameters
    ----------
    combined : dict
        Mapping from model_id to a list of records, each record with keys
        'gold' and 'pred'.
    output_path : str or None, default=None
        Optional path to save the metrics as JSON.

    Returns
    -------
    dict
        Nested dictionary of metrics per model_id, as produced by
        sklearn.metrics.classification_report with output_dict=True.
    """
    all_metrics = {}
    for model_id, results in combined.items():
        y_true = [r["gold"] for r in results]
        y_pred = [r["pred"] for r in results]

        report = classification_report(
            y_true,
            y_pred,
            labels=FULL_LABEL_NAMES,
            zero_division=0,
            output_dict=True
        )
        all_metrics[model_id] = report

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    return all_metrics


def compute_distribution(results):
    """
    Compute gold label distribution for a set of results.

    Parameters
    ----------
    results : list of dict
        Records with at least a 'gold' label.

    Returns
    -------
    total : int
        Number of records.
    dist : dict
        Mapping from label to frequency count.
    """
    total = len(results)
    dist = {label: 0 for label in FULL_LABEL_NAMES}
    for r in results:
        label = r.get('gold')
        if label in dist:
            dist[label] += 1
    return total, dist


def print_report(metrics: dict, total: int, dist: dict):
    """
    Print per-label and macro metrics in tabular form.

    Parameters
    ----------
    metrics : dict
        Metrics dictionary from evaluate().
    total : int
        Number of records.
    dist : dict
        Gold label distribution.
    """
    print(f"Total sections: {total}")
    print("Label distribution (gold):")
    print(f"{'Label':<50s}{'Count':>8s}")
    for label, count in dist.items():
        print(f"{label:<50s}{count:8d}")
    print("\n")

    for model_id, rpt in metrics.items():
        print(f"=== Model: {model_id} ===")
        print(f"{'Label':<50s}{'P':>8s}{'R':>8s}{'F1':>8s}")
        for label in FULL_LABEL_NAMES:
            scores = rpt.get(label, {})
            p = scores.get("precision", 0) * 100
            r = scores.get("recall", 0) * 100
            f = scores.get("f1-score", 0) * 100
            print(f"{label:<50s}{p:8.0f}{r:8.0f}{f:8.0f}")
        mac = rpt.get("macro avg", {})
        p, r, f = mac.get("precision", 0) * 100, mac.get("recall", 0) * 100, mac.get("f1-score", 0) * 100
        print(f"{'Macro average':<50s}{p:8.0f}{r:8.0f}{f:8.0f}")
        print("-" * 74)


def print_delta(metrics1, metrics2):
    """
    Print delta comparison of metrics between two runs.

    Parameters
    ----------
    metrics1 : dict
        Metrics dictionary from evaluate() for file1.
    metrics2 : dict
        Metrics dictionary from evaluate() for file2.
    """
    common_models = set(metrics1.keys()).intersection(metrics2.keys())
    print("\n=== DELTA COMPARISON (File2 - File1) ===")
    for model_id in sorted(common_models):
        print(f"\nModel: {model_id}")
        print(f"{'Label':<50s}{'ΔP':>8s}{'ΔR':>8s}{'ΔF1':>8s}")
        for label in FULL_LABEL_NAMES:
            m1 = metrics1[model_id].get(label, {})
            m2 = metrics2[model_id].get(label, {})
            dp = (m2.get("precision", 0) - m1.get("precision", 0)) * 100
            dr = (m2.get("recall", 0) - m1.get("recall", 0)) * 100
            df = (m2.get("f1-score", 0) - m1.get("f1-score", 0)) * 100
            print(f"{label:<50s}{dp:8.1f}{dr:8.1f}{df:8.1f}")
        m1_mac = metrics1[model_id].get("macro avg", {})
        m2_mac = metrics2[model_id].get("macro avg", {})
        dp = (m2_mac.get("precision", 0) - m1_mac.get("precision", 0)) * 100
        dr = (m2_mac.get("recall", 0) - m1_mac.get("recall", 0)) * 100
        df = (m2_mac.get("f1-score", 0) - m1_mac.get("f1-score", 0)) * 100
        print(f"{'Macro average':<50s}{dp:8.1f}{dr:8.1f}{df:8.1f}")
        print("-" * 74)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and compare LLM results per label")
    parser.add_argument("--input1", required=True, help="Path to first combined_results.json")
    parser.add_argument("--input2", required=True, help="Path to second combined_results.json")
    parser.add_argument("--output1", default=None, help="Optional: path to save first metrics JSON")
    parser.add_argument("--output2", default=None, help="Optional: path to save second metrics JSON")
    parser.add_argument("--log", default="comparsion_log.txt", help="File to write console output to")
    args = parser.parse_args()

    with open(args.log, "w", encoding="utf-8") as logfile:
        sys.stdout = Tee(sys.stdout, logfile)

        with open(args.input1, "r", encoding="utf-8") as f1, open(args.input2, "r", encoding="utf-8") as f2:
            combined1 = json.load(f1)
            combined2 = json.load(f2)

        first_model = next(iter(set(combined1.keys()).intersection(combined2.keys())))

        metrics1 = evaluate(combined1, args.output1)
        metrics2 = evaluate(combined2, args.output2)

        print("=== FILE 1 REPORT ===")
        total, dist = compute_distribution(combined1[first_model])
        print_report(metrics1, total, dist)

        print("=== FILE 2 REPORT ===")
        total, dist = compute_distribution(combined2[first_model])
        print_report(metrics2, total, dist)

        print_delta(metrics1, metrics2)
