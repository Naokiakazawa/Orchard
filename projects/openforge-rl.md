# OpenForge RL

**Train harness-native agents in any environment.**

- 📄 Paper: [*OpenForgeRL: Train Harness-native Agents in Any Environment*](https://arxiv.org/abs/2607.21557) (Yu et al., arXiv:2607.21557)
- 💻 Code: [MSR-Orchard/OpenForge-RL](https://github.com/MSR-Orchard/OpenForge-RL)
- 🗓️ Released 2026-07

## What it adds on top of Orchard

An agent in production is a model *plus a harness* — the scaffold that builds
prompts, parses tool calls, manages history, and retries. Open training stacks
usually train against a simplified reimplementation of that harness, so the
policy you train is not the policy you ship. OpenForge RL removes that mismatch
by training the agent inside the **real** harness against the real environment:
one rollout is a full episode driven by the actual harness, and every model call
along the way becomes training data.

Three pieces sit between Orchard Env and the trainer:

1. **Orchestrator + sandboxes** — each rollout runs in its own Orchard Env
   container, reachable over HTTP, so rollouts scale out while training stays on
   the GPU host.
2. **A recording proxy** — the harness calls what looks like an ordinary
   OpenAI-compatible endpoint; the proxy serves the request and records every
   prompt/response pair. This is what makes an *unmodified* harness trainable.
3. **The trainer** — [veRL](https://github.com/volcengine/verl) consumes the
   recorded episodes for multi-turn RL (GRPO / MM-GRPO / DAPO, Megatron or FSDP).

The result is combinatorial: any harness × any environment, with standard RL
codebases underneath.

## Which Orchard Env features it relies on

- **Remote sandboxes over HTTP** — rollouts run as remote containers rather than
  in-process, decoupling rollout scale-out from the GPU host.
- **Preinstalled harnesses** — `codex`, `claude`, `pi`, `opencode`, and `hermes`
  are already on `PATH` in every sandbox
  ([details](../orchard_env/README.md#built-in-agent-harnesses)), so switching the
  harness you train or evaluate against is a change of command, not a change of
  image.
- **The Python SDK as a git dependency** — `uv sync` installs `orchard_env`
  automatically; OpenForge's extensions over the client live in
  `openforgerl/orchard_compat.py`.

## Environments and modes

Environment code lives in `openforgeenvs/`; the rollout engine in `openforgerl/`
supports two interfacing modes:

- **`PipelineEnvInstanceConfig` (blackbox pipeline, recommended)** — the env owns
  the entire episode: it gets a fresh sandbox plus an `env_run_config`, sets up
  the pod, runs the real harness against `llm_args`, verifies, and returns the
  reward. `claw_eval/` and `osworld/` work this way.
- **`DirectEnvInstanceConfig` (engine-driven loop)** — the engine's own ReACT loop
  steps the env turn by turn, with `prompts.py` / `projection.py` mapping model
  text to env actions. The browser env (`onlinem2w-molmo`) works this way.

Shipped runtimes: `claw_eval` (CLI/tool agents, evaluated on ClawEval), `osworld`
(computer use, evaluated on OSWorld-Verified), and `onlinem2w-molmo` (browser
use, evaluated on Online-Mind2Web / WebVoyager). Tasks are data rather than repo
code, built by a synthesis pipeline to be released separately.

## Headline results

| Model | Benchmark | Score |
| --- | --- | --- |
| OpenForge-GUI (8B) | OSWorld-Verified | **37.7** |
| OpenForge-GUI (8B) | WebVoyager | **72.3** |
| OpenForge-Claw (30B-A3B) | QwenClawBench | **33.7** |
| OpenForge-Claw (30B-A3B) | MCPAtlas | **28.1** |

## Getting started

Point the OpenForge scripts at an Orchard Env orchestrator using the same
environment variables the Orchard client reads:

```bash
export SANDBOX_BASE_URL="http://<orchestrator-host>:80"
export SANDBOX_API_KEY="<key>"
```

No orchestrator yet? See
[Deploying your own cluster](../orchard_env/README.md#deploying-your-own-cluster).
Setup, the patched veRL submodule, and per-domain training/inference scripts are
documented in the [OpenForge-RL README](https://github.com/MSR-Orchard/OpenForge-RL).

> The upstream repository notes it is still being actively reorganized — expect
> in-flight renames and scripts referencing paths that are not yet public.

## Citation

```bibtex
@misc{yu2026openforgerl,
  title={OpenForgeRL: Train Harness-native Agents in Any Environment},
  author={Yu, Xiao and Peng, Baolin and Xu, Ruize and Zou, Hao and Wu, Qianhui and
          Cheng, Hao and Yao, Wenlin and Singh, Nikhil and Yu, Zhou and Gao, Jianfeng},
  year={2026},
  eprint={2607.21557},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2607.21557}
}
```
