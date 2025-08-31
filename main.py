"""
Run C3PA multi-model inference and save combined results.

Usage:
------
python run_experiment.py \
    --models "openai:gpt-4.1-mini,gemini:gemini-2.0-flash" \
    --n_policies 10 \
    --output results.json

Notes:
- API keys are loaded from a `.env` file.
- Supported providers: OpenAI, Anthropic, Google Gemini, Groq, DeepSeek, OpenRouter, Ollama.
"""

import argparse
import logging
import os
import json
from dotenv import load_dotenv

from inference import run_inference


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Run C3PA multi-model inference and save combined results"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="openai:gpt-4.1-mini",
        help=(
            "Comma-separated list of model IDs, e.g. "
            "'openai:gpt-3.5-turbo,openai:gpt-4.1-mini'"
        ),
    )
    parser.add_argument(
        "--n_policies",
        type=int,
        default=4,
        help="Number of policies to sample (default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="all_results.json",
        help="Path to save the JSON file with all models' results",
    )
    args = parser.parse_args()

    # Load .env file
    load_dotenv()
    logging.info("Environment variables loaded from .env")

    for key in [
        "OPENAI_API_KEY",
        "GROQ_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "OLLAMA_API_URL",
    ]:
        if os.getenv(key) is None:
            logging.warning(f"Environment variable {key} is not set.")

    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    logging.info(f"Models to evaluate: {model_list}")

    # Run inference
    combined = run_inference(models=model_list, n_policies=args.n_policies, )

    # Save combined results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved results for {len(model_list)} models to {args.output}")


if __name__ == "__main__":
    main()
