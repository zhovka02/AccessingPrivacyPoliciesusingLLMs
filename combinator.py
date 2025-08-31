"""
Utilities to merge JSON result files from multiple models and report coverage.

This module provides:
- `combine_results`: Merge result dictionaries keyed by `model_id`, deduplicate
  records per model based on `(policy_id, span, gold, pred)`, and optionally
  write the combined output to disk.
- `count_unique_policies`: Count distinct `policy_id`s per model in the
  combined structure.
- A CLI that accepts multiple input JSON files and writes a combined file while
  printing per-model totals and unique policy counts.

Expected input JSON schema per file:
{
  "<model_id>": [
    {
      "policy_id": "<string|int>",
      "span": "<string>",
      "gold": "<string>",
      "pred": "<string>",
      ...
    },
    ...
  ],
  ...
}

The combined structure preserves all fields of each record, only using the
four-tuple `(policy_id, span, gold, pred)` for deduplication.
"""

import json
import argparse
from typing import Dict, List, Iterable, Optional, Any


def combine_results(input_paths: Iterable[str], output_path: Optional[str] = None) -> Dict[str, List[dict]]:
    """
    Merge multiple model result JSON files into a single dictionary keyed by model ID.

    Records are deduplicated per model using the key:
    `(policy_id, span, gold, pred)`. If `output_path` is provided, the combined
    dictionary is written as pretty-printed JSON with UTF-8 encoding.

    Parameters
    ----------
    input_paths : Iterable[str]
        Paths to input JSON files. Each file must map model IDs to a list of records.
    output_path : Optional[str], default=None
        Destination file path for the combined JSON. If None, the result is not written.

    Returns
    -------
    Dict[str, List[dict]]
        Combined and deduplicated results keyed by model ID.

    Raises
    ------
    json.JSONDecodeError
        If any input file contains invalid JSON.
    FileNotFoundError
        If an input path does not exist.
    """
    combined: Dict[str, List[dict]] = {}
    for path in input_paths:
        with open(path, 'r', encoding='utf-8') as f:
            data: Dict[str, List[dict]] = json.load(f)
        for model_id, records in data.items():
            if model_id not in combined:
                combined[model_id] = []
            seen = {
                (r.get('policy_id'), r.get('span'), r.get('gold'), r.get('pred'))
                for r in combined[model_id]
            }
            for r in records:
                key = (r.get('policy_id'), r.get('span'), r.get('gold'), r.get('pred'))
                if key not in seen:
                    combined[model_id].append(r)
                    seen.add(key)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
    return combined


def count_unique_policies(combined: Dict[str, List[dict]]) -> Dict[str, int]:
    """
    Compute the number of distinct `policy_id`s per model.

    Parameters
    ----------
    combined : Dict[str, List[dict]]
        Combined results structure as returned by `combine_results`.

    Returns
    -------
    Dict[str, int]
        Mapping from model ID to count of unique `policy_id` values.
    """
    unique_counts: Dict[str, int] = {}
    for model_id, records in combined.items():
        policy_ids = {r.get('policy_id') for r in records if 'policy_id' in r}
        unique_counts[model_id] = len(policy_ids)
    return unique_counts


def main() -> None:
    """
    Command-line entry point.

    Usage
    -----
    python script.py --inputs file1.json file2.json ... [--output combined.json]

    Prints per-model totals and counts of unique policy IDs, and writes the
    combined JSON if `--output` is provided (default: `combined.json`).
    """
    parser = argparse.ArgumentParser(
        description="Combine multiple model result JSON files, deduplicate records, and report unique policy counts per model."
    )
    parser.add_argument(
        "--inputs", nargs='+', required=True,
        help="List of input JSON files to combine."
    )
    parser.add_argument(
        "--output", default="combined.json",
        help="Output file path for the combined JSON (default: combined.json)."
    )
    args = parser.parse_args()

    combined = combine_results(args.inputs, args.output)
    unique_counts = count_unique_policies(combined)

    print("Combined results with deduplication and unique policy_id counts per model:")
    for model_id, records in combined.items():
        total = len(records)
        unique = unique_counts.get(model_id, 0)
        print(f"Model: {model_id}\n  Total unique records: {total}\n  Unique policy_ids: {unique}\n{'-'*40}")


if __name__ == "__main__":
    main()