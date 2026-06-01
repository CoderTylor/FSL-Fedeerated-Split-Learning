# FSLT GitHub Code

Federated Split Learning Testbed (FSLT) socket implementation with three roles:
UE clients, split-learning server, and federated aggregation server.

## Architecture

```text
UE Clients
  ├─ run client-side model
  ├─ send activations to the split server
  ├─ receive gradients and update local client models
  └─ upload client-side model weights for aggregation

Split-Learning Server
  ├─ receives UE activations
  ├─ runs server-side model forward/backward pass
  └─ returns activation gradients to UEs

Federated Aggregation Server
  ├─ receives updated UE-side model weights
  ├─ performs FedAvg
  └─ sends the global UE-side model back for the next round
