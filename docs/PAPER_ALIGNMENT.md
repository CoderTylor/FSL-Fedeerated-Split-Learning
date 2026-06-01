# Paper Alignment Notes

This document records the main mismatches found in the original folder and how
this cleaned project addresses them.

## Fixed In This Clean Project

| Issue in original folder | Why it was weak | Change in this project |
|---|---|---|
| Hard-coded host `10.37.6.90`, ports, clients, and epochs | Hard to reproduce on another machine or VM topology | Added shared CLI config in `config.py` |
| `multi_client.py` used a fixed `time.sleep(5)` before downloading the global model | Could start too early or wait longer than needed | Client now retries the aggregation download port until the server is ready |
| Next round was implicit | Paper algorithm requires aggregation and redistribution before the next epoch/round | Client now loads the global model before continuing, then applies optional `--round-delay-seconds` |
| No UE selection logic | Paper discusses FSLT-select and random UE participation | Added `all`, `random`, and latency-based selection in `round_control.py` |
| No communication metric logging | Paper reports latency and packet-size behavior | Added JSONL logging for bytes, split latency, aggregation wait, and server service time |
| Socket receive loop depended on shutdown/end-of-stream | Fragile for larger messages and future reuse | Added length-prefixed protocol in `protocol.py` |
| Evaluation used a fresh local server-side model | Printed accuracy was not the actual split model accuracy | Removed that misleading evaluation from the main client loop |
| Mixed active code, old baselines, data, binaries, and pycache | Not appropriate for GitHub | New folder includes only related project code and docs |

## Still Not Exact Paper Reproduction

The paper describes:

- OpenAirInterface/RFsim 5G integration.
- Rayleigh fading/channel settings and packet-loss measurement.
- Avatar skeleton extraction, not HAM10000 skin lesion classification.
- PointConv-style split layers and AP/MSE-based evaluation.
- Up to 10 candidate UE VMs and FSLT-select based on observed channel quality.

This repository demonstrates the same distributed learning control flow using a
clean TCP socket prototype. To make it the full paper artifact, the next
engineering step is to connect the metric hooks to OAI/RFsim measurements and
replace the HAM10000/ResNet example with the skeleton extraction model and
dataset used in the manuscript.
