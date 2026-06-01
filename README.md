# FSLT GitHub Code

This is a cleaned, GitHub-ready Federated Split Learning Testbed (FSLT) socket
prototype. It keeps only code relevant to the UE client, split-learning server,
and federated aggregation server workflow.

No dataset files are included.

## What This Repository Contains

```text
src/fslt_socket/
├── aggregation.py      # Federated server and FedAvg
├── client.py           # UE client training loop
├── config.py           # Shared CLI/configuration
├── data_loader.py      # Local HAM10000 loader, no bundled data
├── metrics.py          # JSONL metric logging
├── model.py            # Split ResNet18 client/server model
├── protocol.py         # Length-prefixed socket protocol
├── round_control.py    # UE selection and inter-round delay
└── split_server.py     # Split-learning server
```

The old single-file baselines and raw datasets were intentionally left out of
this folder. They are useful as historical reference, but they make the GitHub
project noisy and blur the code path used by the FSLT socket demo.

## Main Improvements Over The Original Folder

- Removed datasets, archives, `.DS_Store`, `__pycache__`, unrelated binaries,
  and old draft scripts.
- Replaced hard-coded script constants with CLI arguments.
- Replaced ad-hoc socket reads with a length-prefixed protocol.
- Added a real round barrier: clients upload selected models, wait for the
  aggregated global model, load it, and only then start the next round.
- Added configurable delay before the next round via `--round-delay-seconds`.
- Added optional UE selection strategies: `all`, `random`, and `latency`.
- Added JSONL metric logging for latency, bytes sent/received, aggregation
  wait time, and server loss.
- Removed misleading local evaluation that used an untrained local server-side
  model.
- Added unit tests for FedAvg, socket message encoding, UE selection, and
  round delay behavior.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Dataset

Prepare HAM10000 locally:

```text
data/HAM10000_metadata.csv
data/HAM10000_images_part_1/*.jpg
data/HAM10000_images_part_2/*.jpg
```

The dataset is ignored by Git. Keep it local unless you have checked the
redistribution terms.

## Run

Open three terminals from the repository root.

Terminal 1, split-learning server:

```bash
python -m fslt_socket.split_server --host 127.0.0.1
```

Terminal 2, federated aggregation server:

```bash
python -m fslt_socket.aggregation --host 127.0.0.1
```

Terminal 3, UE clients:

```bash
python -m fslt_socket.client \
  --host 127.0.0.1 \
  --num-clients 3 \
  --clients-per-round 3 \
  --epochs 10 \
  --round-delay-seconds 1.0 \
  --client-selection latency
```

For your OAI/RFsim VM setup, replace `127.0.0.1` with the reachable split
server / aggregation server IP address.

## Useful Options

- `--host`: server address used by all three roles.
- `--activation-port`: UE to split-server activation port.
- `--aggregation-upload-port`: UE model update upload port.
- `--aggregation-download-port`: global model download port.
- `--clients-per-round`: number of UEs selected in each federated round.
- `--client-selection`: `all`, `random`, or `latency`.
- `--round-delay-seconds`: wait time after aggregation before the next round.
- `--metrics-path`: output JSONL metric path.
- `--metadata-path` and `--image-glob`: local HAM10000 dataset locations.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

## Paper Alignment

This folder is now aligned with the paper's high-level Algorithm 1 control
flow: UE-side forward propagation, split-server propagation/backpropagation,
UE-side update, federated aggregation, global model distribution, and delayed
next-round start.

It is still a socket prototype, not a complete OAI/RFsim artifact. See
[docs/PAPER_ALIGNMENT.md](docs/PAPER_ALIGNMENT.md) for what was fixed and what
would still be needed for exact paper reproduction.
