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

### `orchard_eval/` — evaluation

Runs **any** agent harness against a benchmark on Orchard Env sandboxes: a fresh
sandbox for the agent, a second fresh one for grading, and scoring by the
benchmark's own harness. The benchmark, the environment, and the agent are kept
as separate layers, so changing the agent under test is a config edit rather
than a new runner.

- **Suite** (`orchard_eval/orchard_evalkit/`) — the importable package; note it
  is named differently from the directory that holds it. `datasets/` and
  `grading/` own the benchmark (SWE-bench Verified, Multilingual and Pro),
  `sandbox.py` / `runner.py` own the pods, and `harnesses/` owns the agent loop
  — `codex`, `claude`, `opencode`, `pi` and `mini-swe-agent`, plus gold-patch
  and no-patch harnesses for calibrating a run before spending on it.
- **Harbor bridge** (`orchard_evalkit/harbor_bridge.py`) — drives `harbor run`
  for the Harbor-format benchmarks (Terminal-Bench 2.1, DeepSWE 1.1, SWE-bench
  Pro) instead of reimplementing their trial semantics, then re-reads the
  results into the same shape the rest of the suite reports. SWE-bench Pro is
  reachable both ways — natively through `datasets/`+`grading/` and through
  Harbor — which is what makes each number checkable against the other.
- **Harbor provider** (`orchard_eval/harbor_orchard/`) — a separately
  installable package (`harbor-orchard`) that Harbor loads by import path as
  `harbor_orchard:OrchardEnvironment`. It turns a task's Dockerfile into
  commands and its mounts into transfers, which is what lets a Harbor task —
  which expects a local Docker build and a bind mount — run on a pod instead.
- **CLI** — `orchard-eval run | harbor | report | list-harnesses`, with the
  benchmark/harness combinations declared in `orchard_eval/configs/`.

See [orchard_eval/README.md](../orchard_eval/README.md) and
[orchard_eval/harbor_orchard/README.md](../orchard_eval/harbor_orchard/README.md).

### `trainer/slime/` — the trainer

A fork of the [slime](https://github.com/THUDM/slime) RL training stack, vendored
as a git submodule pointing at
[MSR-Orchard/slime](https://github.com/MSR-Orchard/slime). Orchard-specific
rollout code lives under
[`examples/orchard_swe/`](https://github.com/MSR-Orchard/slime/tree/main/examples/orchard_swe)
and
[`examples/orchard_gui/`](https://github.com/MSR-Orchard/slime/tree/main/examples/orchard_gui).

Because it is a submodule, a plain `git clone` leaves `trainer/slime/` empty —
clone with `--recursive`, or run `git submodule update --init trainer/slime`.

Unlike the environment layer, the trainer requires GPUs. See
[Training](../README.md#training) for the node specs.

## How they fit together

```
   ┌──────────────┐    rollout requests    ┌────────────────────┐
   │  trainer/    │ ─────────────────────▶ │                    │
   │  slime       │ ◀───────────────────── │   orchard_env      │
   └──────────────┘   trajectories/rewards │   orchestrator     │
                                           │   (FastAPI)        │
   ┌──────────────┐   one sandbox per task │                    │
   │ orchard_eval │ ─────────────────────▶ │                    │
   │  + harbor    │ ◀───────────────────── │                    │
   └──────────────┘    patches / rewards   └─────────┬──────────┘
                                                     │ HTTP to pod IP
                                           ┌─────────▼──────────┐
                                           │   sandbox pods     │
                                           │   (in-pod agent)   │
                                           └────────────────────┘
```

Both drive the same SDK. The trainer asks for rollouts and gets one sandbox per
rollout, with results flowing back as trajectories for the RL loop;
`orchard_eval` asks for one sandbox per benchmark instance and a second for
grading. Because the substrate is identical, a model is measured under the same
execution conditions it was trained in.
