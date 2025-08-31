"""
Span filtering utility for C3PA annotation CSVs.

This script scans annotation files, removes conflicting spans where:
- The text similarity between two spans (by difflib.SequenceMatcher) is >= THRESHOLD
- The gold labels differ

Both spans in each conflicting pair are removed.

Filtered CSVs are written to a separate directory, preserving filenames.
"""

import os
import pandas as pd
from difflib import SequenceMatcher

# Configuration
DATA_ROOT = "/Users/aleksey/PycharmProjects/AccessingPPusingLLM/C3PA_Dataset-main"
ANNOT_DIR = os.path.join(DATA_ROOT, "Annotations/DB")
FILTERED_DIR = os.path.join(DATA_ROOT, "Filtered/DB")
THRESHOLD = 0.75  # similarity threshold

os.makedirs(FILTERED_DIR, exist_ok=True)


def filter_spans(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Remove conflicting spans based on similarity and differing labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least columns 'Text' and 'Label'.
    threshold : float
        Similarity ratio (0–1). If >= threshold and labels differ, both spans are removed.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with reset index.
    """
    to_drop = set()
    texts = df['Text'].astype(str).tolist()
    labels = df['Label'].astype(str).tolist()
    n = len(df)

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if ratio >= threshold:
                    to_drop.update([i, j])

    return df.drop(index=to_drop).reset_index(drop=True)


def process_all_files(annot_dir: str, out_dir: str, threshold: float) -> None:
    """
    Process all CSV files in a directory, filtering spans and saving results.

    Parameters
    ----------
    annot_dir : str
        Directory containing annotation CSVs.
    out_dir : str
        Directory to save filtered CSVs.
    threshold : float
        Similarity threshold for filtering.
    """
    for filename in os.listdir(annot_dir):
        if not filename.endswith(".csv"):
            continue
        out_path = os.path.join(out_dir, filename)
        if os.path.exists(out_path):
            print(f"[SKIP] {filename} (already filtered)")
            continue
        csv_path = os.path.join(annot_dir, filename)
        df = pd.read_csv(csv_path)

        filtered_df = filter_spans(df, threshold)
        filtered_df.to_csv(out_path, index=False)

        print(f"[DONE] {filename}: {len(df)} → {len(filtered_df)} spans")

    print("All files processed. Filtered CSVs saved to:", out_dir)


if __name__ == "__main__":
    process_all_files(ANNOT_DIR, FILTERED_DIR, THRESHOLD)
