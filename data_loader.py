"""
Policy dataset loader and context extractor for the C3PA dataset.

This module provides functions to:
- Extract visible text from HTML files representing privacy policies.
- Clean and normalize text spans for alignment.
- Align annotated spans with their surrounding context sentences.
- Load a configurable sample of annotated policies either from raw
  annotations (HTML + CSV) or from precomputed JSON dumps with context.

The dataset structure is expected as in the official C3PA release, with
subdirectories:
- `Annotations/WS`  : CSV files with span annotations and labels.
- `Htmls/WS`        : Raw HTML source files for policies.
- `Texts/DB`        : Derived plain-text versions of HTML policies.
- `Contexts/WS|DB`  : JSON dumps containing text, annotations, and context.

The loader supports reproducible random sampling of policies and annotation
alignment with up to two sentences of left context.
"""

import json
import os
import random
import pandas as pd
import re
from typing import List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup


DATA_ROOT = "/Users/aleksey/PycharmProjects/AccessingPPusingLLM/C3PA_Dataset-main"


def extract_visible_text_from_html(html: str) -> str:
    """
    Extract visible text content from an HTML document.

    Removes <script> and <style> blocks and returns visible text
    with whitespace normalized.

    Parameters
    ----------
    html : str
        Raw HTML string.

    Returns
    -------
    str
        Cleaned visible text.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return text


def remove_all_non_chars(s: str) -> str:
    """
    Remove all non-alphanumeric characters from a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    str
        String containing only alphanumeric characters.
    """
    return ''.join(e for e in s if e.isalnum())


def find_sentence_indices_for_span(span: str, sentences: List[str]) -> int:
    """
    Identify the index of the sentence in which a span most likely begins.

    Heuristically cleans the span, derives an anchor token sequence,
    and searches for a matching sentence index.

    Parameters
    ----------
    span : str
        Annotated text span.
    sentences : List[str]
        List of sentence strings from the full policy text.

    Returns
    -------
    int or None
        Index of the first sentence containing the span anchor,
        or None if no match is found.
    """
    if isinstance(span, float):
        return None

    span_clean = span.replace("\\n", "").replace("\\t", "")
    span_nows = remove_all_non_chars(span_clean)
    if not span_nows:
        return None

    sentences_span = sent_tokenize(span_clean)

    if sentences_span:
        first_piece = re.split(r'[^\w\s]', sentences_span[0], maxsplit=1)[0]
        clean_first = remove_all_non_chars(first_piece)
        if len(clean_first) < 30:
            anchor_first = clean_first
        else:
            anchor_first = clean_first[:30]
    else:
        anchor_first = span_nows[:30]

    index_first = None

    for i, sent in enumerate(sentences):
        sent_clean = remove_all_non_chars(sent)
        if index_first is None and anchor_first in sent_clean:
            index_first = i
            break

    return index_first


def load_policies(
    n_policies: int = 20,
    random_seed: int = 21,
    with_context: bool = True,
    use_ready_data: bool = True
) -> List[Dict[str, Any]]:
    """
    Load a random sample of annotated privacy policies.

    Supports loading from precomputed JSON dumps with context
    (fast path) or from raw HTML + annotation CSV files
    (slow path, with context extraction).

    Parameters
    ----------
    n_policies : int, default=20
        Number of policies to sample.
    random_seed : int, default=21
        Random seed for reproducible sampling.
    with_context : bool, default=True
        Whether to align annotations with left context sentences.
    use_ready_data : bool, default=True
        If True, load from JSON dumps with context.
        If False, rebuild from raw annotations and HTML.

    Returns
    -------
    List[Dict[str, Any]]
        List of policies, each containing:
        - 'policy_id' : str
        - 'text' : str
        - 'annotations' : list of dicts with keys {Text, Label, Context}
    """
    random.seed(random_seed)
    if use_ready_data:
        return load_ready_data_with_context(n_policies, random_seed)

    annot_dir = os.path.join(DATA_ROOT, "Annotations/WS")
    all_csvs = [os.path.join(annot_dir, f) for f in os.listdir(annot_dir) if f.endswith(".csv")]
    policies_no_context = []
    all_csvs = [f for f in all_csvs if policies_no_context.count(os.path.basename(f)) == 0]
    chosen = random.sample(all_csvs, n_policies)
    dataset = []

    for csv_path in chosen:
        df = pd.read_csv(csv_path)
        policy_id = os.path.splitext(os.path.basename(csv_path))[0]

        html_path = os.path.join(DATA_ROOT, "Htmls/WS", policy_id + ".html")
        with open(html_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        full_text = extract_visible_text_from_html(raw_html)

        if with_context:
            text_file_path = os.path.join(DATA_ROOT, "Texts/DB", f"{policy_id}.txt")
            os.makedirs(os.path.dirname(text_file_path), exist_ok=True)
            with open(text_file_path, "w", encoding="utf-8") as text_file:
                text_file.write(full_text)

            sentences = sent_tokenize(full_text)

            annotations = []
            for record in df.to_dict(orient="records"):
                raw_span = record.get("Text", "")
                label = record.get("Label", "").strip()
                sent_index_first = find_sentence_indices_for_span(raw_span, sentences)

                context = ""
                if sent_index_first is not None:
                    start_idx = max(0, sent_index_first - 2)
                    context = "2 Sentences Before {" + " ".join(
                        sentences[start_idx:sent_index_first]) + "} "
                annotations.append({
                    "Text": raw_span,
                    "Label": label,
                    "Context": context
                })
        else:
            annotations = []
            for record in df.to_dict(orient="records"):
                raw_span = record.get("Text", "")
                label = record.get("Label", "").strip()
                annotations.append({
                    "Text": raw_span,
                    "Label": label,
                    "Context": ""
                })
        dataset.append({
            "policy_id": policy_id,
            "text": full_text,
            "annotations": annotations
        })

    return dataset


def load_ready_data_with_context(
    n_policies: int = 20,
    random_seed: int = 21
) -> List[Dict[str, Any]]:
    """
    Load a random sample of policies from precomputed JSON dumps with context.

    Parameters
    ----------
    n_policies : int, default=20
        Number of policies to sample.
    random_seed : int, default=21
        Random seed for reproducible sampling.

    Returns
    -------
    List[Dict[str, Any]]
        List of preprocessed policy dictionaries with text and annotations.
    """
    random.seed(random_seed)

    context_dir = os.path.join(DATA_ROOT, "Contexts", "WS")
    all_files = [f for f in os.listdir(context_dir) if f.endswith('.json')]
    if len(all_files) < n_policies:
        raise ValueError(f"Es sind nur {len(all_files)} Policy-Dumps verfügbar, aber {n_policies} angefordert.")
    selected_files = random.sample(all_files, n_policies)

    data: List[Dict[str, Any]] = []
    for filename in selected_files:
        dump_file_path = os.path.join(context_dir, filename)
        with open(dump_file_path, "r", encoding="utf-8") as f:
            entry = json.load(f)
        data.append(entry)

    return data


if __name__ == "__main__":
    """
    Script entry point.

    Loads a sample of policies (default: 105), extracts contexts if needed,
    and dumps them to JSON files in 'Contexts/DB`.
    """
    data = load_policies(n_policies=105, random_seed=12, use_ready_data=False)


    for entry in data:
        dump_file_path = os.path.join(DATA_ROOT, "Contexts/DB", f"{entry['policy_id']}.json")
        os.makedirs(os.path.dirname(dump_file_path), exist_ok=True)
        with open(dump_file_path, "w", encoding="utf-8") as text_file:
            json.dump(entry, text_file, indent=2, ensure_ascii=False)
