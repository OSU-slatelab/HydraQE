# [HydraQE](https://aclanthology.org/2026.iwslt-1.37/)

This repository contains inference code for HydraQE, an end-to-end, reference-free speech translation quality estimator built on Qwen3-ASR.

The model was trained on out-of-English translation directions; the speech encoder supports other languages but has not been evaluated on them.

## Python API

```python
from hydraqe import HydraQE

model = HydraQE.from_pretrained(
    "osu-slatelab/HydraQE",
    device="cuda",
    dtype="bfloat16",
)

scores = model.score(
    audio="source.wav",
    translation="Die zu bewertende Systemübersetzung.",
)
print(scores)
# {
#   "da": ...,
#   "metricx_ref": ...,
#   "xcomet_ref": ...,
# }
```

All scores use a nominal 0–100 scale and higher is better. Values are not
clipped. To request a subset:

```python
scores = model.score("source.wav", "...", heads=["da"])
```

Batch inference accepts audio paths, NumPy arrays, Torch tensors, or
`(waveform, sample_rate)` tuples:

```python
scores = model.score_batch(
    audio=["one.wav", "two.wav"],
    translations=["Erste Übersetzung", "Zweite Übersetzung"],
)
```

Bare arrays and tensors are assumed to be mono 16 kHz audio. Files and explicit
sample-rate tuples are resampled to 16 kHz. Audio longer than 30 seconds is truncated.

## Command line

Input JSONL rows must contain `audio_path` and either `translation` or
`tgt_text`:

```json
{"audio_path":"audio/example.wav","translation":"候选译文"}
```

```bash
hydraqe \
  --input examples.jsonl \
  --output predictions.jsonl \
  --head xcomet_ref
```

`--head` accepts `da`, `metricx_ref`, `xcomet_ref`, or `mean`. The `mean`
option averages all three heads. Each output row preserves the input fields and
adds the selected scalar under `score`:

```json
{"audio_path":"audio/example.wav","translation":"候选译文","score":72.4}
```

## Reference
If you use HydraQE in your research, please cite:
```
@inproceedings{krahn-fosler-lussier-2026-hydraqe,
    title = "{H}ydra{QE}: {OSU}{'}s Submission for the {IWSLT} 2026 Speech Translation Metrics Shared Task",
    author = "Krahn, Kevin  and
      Fosler-Lussier, Eric",
    editor = "Salesky, Elizabeth  and
      Anastasopoulos, Antonios  and
      Negri, Matteo  and
      Federico, Marcello",
    booktitle = "Proceedings of the 23rd International Conference on Spoken Language Translation ({IWSLT} 2026)",
    month = jul,
    year = "2026",
    address = "San Diego, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.iwslt-1.37/",
    doi = "10.18653/v1/2026.iwslt-1.37",
    pages = "323--331",
    ISBN = "979-8-89176-411-8",
}
```