import argparse
import json
from pathlib import Path

from .inference import HydraQE


def _chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    parser = argparse.ArgumentParser(
        description="Score speech-translation hypotheses with HydraQE"
    )
    parser.add_argument(
        "--model", default="osu-slatelab/HydraQE", help="HF model ID or local model directory"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL")
    parser.add_argument(
        "--head",
        required=True,
        choices=("da", "metricx_ref", "xcomet_ref", "mean"),
        help="Prediction head to export, or mean for the mean over all heads",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--no-merge-lora", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    model = HydraQE.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        merge_lora=not args.no_merge_lora,
    )
    output_rows = []
    for batch in _chunks(rows, args.batch_size):
        paths = [row.get("audio_path") for row in batch]
        translations = [row.get("translation", row.get("tgt_text")) for row in batch]
        if any(path is None for path in paths):
            raise ValueError("Every input row must contain audio_path")
        if any(text is None for text in translations):
            raise ValueError("Every input row must contain translation or tgt_text")
        selected_heads = None if args.head == "mean" else [args.head]
        scores = model.score_batch(paths, translations, heads=selected_heads)
        for index, row in enumerate(batch):
            if args.head == "mean":
                score = sum(values[index] for values in scores.values()) / len(scores)
            else:
                score = scores[args.head][index]
            output_rows.append({**row, "score": score})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows)
    )


if __name__ == "__main__":
    main()
