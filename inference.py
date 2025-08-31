"""
Parallel inference runner for C3PA dataset using multiple LLM backends.

This script:
- Loads annotated policies (with or without context).
- Spawns one thread per model ID.
- Runs classification for each annotation span using a shared system prompt
  and a user prompt containing the span (+ optional context).
- Collects predictions per model and saves intermediate + final dumps to JSON.

Example:
--------
python inference.py \
    --models "openai:gpt-4.1-mini,gemini-2.0-flash" \
    --n_policies 20
"""

import json
import logging
import threading
from typing import List, Dict

from data_loader import load_policies
from llm_client import LLMClient
from promt_templates import SYSTEM_PROMPT, USER_PROMPT

DUMP_PATH = "dump_all.json"


def _infer_for_model(
    model_id: str,
    data: List[Dict],
    all_results: Dict[str, List[Dict]],
    lock: threading.Lock
) -> None:
    """
    Worker function: run inference for one model over all policies.

    Parameters
    ----------
    model_id : str
        Identifier of the LLM backend (e.g. "openai:gpt-4.1-mini").
    data : List[Dict]
        Policies with annotations, as returned by `load_policies`.
    all_results : Dict[str, List[Dict]]
        Shared dictionary (thread-safe via lock) to store results.
    lock : threading.Lock
        Synchronization primitive to safely update shared state.

    Side effects
    ------------
    - Updates `all_results[model_id]` with predictions.
    - Overwrites the shared dump file `DUMP_PATH` after finishing.
    """
    logging.info(f"[{model_id}] Starting inference thread.")
    client = LLMClient(model=model_id)
    results: List[Dict] = []

    for item in data:
        pid = item["policy_id"]
        for ann in item["annotations"]:
            span = ann["Text"]
            gold = ann["Label"]
            context = ann.get("Context", "")

            user_prompt = USER_PROMPT.format(span=span, context=context)
            pred = client.classify(SYSTEM_PROMPT, user_prompt)

            results.append({
                "policy_id": pid,
                "span": span,
                "gold": gold,
                "pred": pred,
            })

            logging.debug(f"[{model_id}] {pid}: gold='{gold}', pred='{pred[:50]}...'")

    with lock:
        all_results[model_id] = results
        with open(DUMP_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    logging.info(f"[{model_id}] Finished; wrote {len(results)} predictions.")


def run_inference(models: List[str], n_policies: int = 20) -> Dict[str, List[Dict]]:
    """
    Run inference for multiple models in parallel (one thread per model).

    Parameters
    ----------
    models : List[str]
        List of model identifiers.
    n_policies : int, default=20
        Number of policies to load for inference.

    Returns
    -------
    Dict[str, List[Dict]]
        Mapping from model_id to list of prediction records.
    """
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO
    )

    logging.info(f"Loading {n_policies} policies...")
    print("dd")
    data = load_policies(n_policies, random_seed=12, with_context=False, use_ready_data=True)
    logging.info("Data loaded. Starting threads for models: " + ", ".join(models))

    all_results: Dict[str, List[Dict]] = {}
    lock = threading.Lock()

    threads = []
    for model_id in models:
        t = threading.Thread(
            target=_infer_for_model,
            args=(model_id, data, all_results, lock),
            name=f"Thread-{model_id}"
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    logging.info("All model threads complete.")
    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run multi-model C3PA inference in parallel threads"
    )
    parser.add_argument(
        "--models", type=str, required=True,
        help="Comma-separated list of model IDs, e.g. 'openai:gpt-4.1-mini,gemini-2.0-flash'"
    )
    parser.add_argument(
        "--n_policies", type=int, default=20,
        help="Number of policies to sample (default: 20)"
    )
    args = parser.parse_args()

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    combined = run_inference(models=model_list, n_policies=args.n_policies)

    print(f"Final combined results saved to {DUMP_PATH}")
