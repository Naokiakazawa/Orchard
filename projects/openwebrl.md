# OpenWebRL

**Online multi-turn RL for visual web agents on live websites.**

- 📄 Paper: [*OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents*](https://arxiv.org/abs/2606.02031) (arXiv:2606.02031)
- 💻 Code: [OpenWebRL/OpenWebRL](https://github.com/OpenWebRL/OpenWebRL)
- 🗓️ Released 2026-06

## What it adds on top of Orchard

OpenWebRL extends Orchard-GUI into a full online multi-turn RL study on *live*
websites, covering supervised initialization, multimodal context management,
trajectory-level success judging, and multi-turn policy optimization. Training on
the live web means the environment itself is unreliable, so the project builds a
fault-tolerant browser environment on Orchard Env — navigation retries, timeout
handling, and structured failure attribution — that keeps unstable website
behavior separable from model behavior at training scale.

The stack sits on a modified `slime` (Megatron / SGLang) trainer and adds the
browser rollout, reward, data, and evaluation components a web agent needs:
Playwright-based interaction, multi-turn multimodal rollouts, tool-call parsing,
textual environment feedback, and VLM-as-a-judge rewards for Qwen3-VL style
models.

## Which Orchard Env features it relies on

- **Large-scale parallel rollouts** — Orchard Env supplies network-isolated
  browser instances on demand, so many live-web episodes run concurrently. (A
  local-process mode is also supported for smaller runs.)
- **Network policy per sandbox** — deny-egress-by-default isolation with
  per-sandbox CPU / memory / timeout limits and TTL cleanup.

## Headline results

| Model | Benchmark | Score |
| --- | --- | --- |
| OpenWebRL-4B | Online-Mind2Web | **67.0%** |
| OpenWebRL-4B | DeepShop | **64.0%** |

From only 0.4K initialization trajectories and 2.2K open-ended RL tasks — a new
open-source state of the art on live-web benchmarks, competitive with OpenAI CUA
and Gemini CUA.

## Getting started

Pipeline stages, training launchers, and evaluation scripts are documented in the
[OpenWebRL README](https://github.com/OpenWebRL/OpenWebRL). To use Orchard Env
for rollouts, stand up an orchestrator
([deployment guide](../orchard_env/README.md#deploying-your-own-cluster)) and
point the client at it:

```bash
export SANDBOX_BASE_URL="http://<orchestrator-host>:80"
export SANDBOX_API_KEY="<key>"
```
