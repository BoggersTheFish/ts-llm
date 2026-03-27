# TinyStories data (local only — not in Git)

The Git repository **does not** include:

- `TinyStories_all_data.tar.gz` (~1.6GB+)
- `extracted/*.json` shards (~140MB each)

Download the official archive, place it under `data/tinystories/`, extract JSON into `data/tinystories/extracted/`, then use `run_attractor_llm.py` with `--dataset tinystories` or run `scripts/prepare_tinystories_tokenized.py` to refresh `processed/*.npy`.

See [DEV_ROUTE_1.md](../../DEV_ROUTE_1.md) for bounded CPU training examples.
