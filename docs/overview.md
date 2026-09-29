# Orchard-Agentic overview

Orchard-Agentic is a collection of open-source research, infrastructure,
datasets, training recipes, and evaluation artifacts for agentic modeling. It
hosts the foundational paper *Orchard: An Open-Source Agentic Modeling
Framework* and subsequent projects such as OpenWebRL and OpenForge RL.

Orchard Env, Orchard-SWE, Orchard-GUI, and Orchard-Claw retain the established
names introduced by the Orchard research. In this repository, Orchard-Agentic
refers to the broader collection, while Orchard refers to the original paper and
framework.

## Components

### `orchard_env/` — the environment

A Kubernetes-based sandbox orchestration service and a Python SDK. It gives an
agent a fresh container per task and lets it run many turns of commands, file
edits, and git patches against that container.

- **Control plane** (`orchard_env/orchestrator/`) — FastAPI service that manages
  sandbox lifecycle, exec jobs, and cleanup. Scales horizontally with Redis.
- **In-pod agent** (`orchard_env/agent/`) — a small FastAPI server injected into
  every sandbox pod. The orchestrator calls it directly over the pod IP, so exec
  and file I/O never touch the Kubernetes API server.
- **SDK** (`orchard_env/client/`) — sync (`SandboxClient`) and async
  (`AsyncSandboxClient`) clients with context-manager lifecycle handling.

See [orchard_env/README.md](../orchard_env/README.md) and
[orchard_env/docs/architecture.md](../orchard_env/docs/architecture.md).

### `trainer/slime/` — the trainer

A vendored fork of the [slime](https://github.com/THUDM/slime) RL training stack,
with Orchard-specific rollout code under `examples/orchard/`. Fork-local changes
are tracked in `trainer/slime/ORCHARD_CHANGES.md`.

## How they fit together

```
   ┌──────────────┐    rollout requests    ┌────────────────────┐
   │  trainer/    │ ─────────────────────▶ │   orchard_env      │
   │  slime       │                        │   orchestrator     │
   │              │ ◀───────────────────── │   (FastAPI)        │
   └──────────────┘   trajectories/rewards └─────────┬──────────┘
                                                     │ HTTP to pod IP
                                           ┌─────────▼──────────┐
                                           │   sandbox pods     │
                                           │   (in-pod agent)   │
                                           └────────────────────┘
```

The trainer drives rollouts through the `orchard_env` SDK; each rollout gets its
own sandbox, and results flow back as trajectories for the RL loop.
