# Projects built on Orchard

Orchard Env is a general substrate, not a training stack: it exposes sandbox
lifecycle, exec, file I/O, and network policy over a REST API, and makes no
assumption about the harness, trainer, inference backend, or task domain sitting
above it. This folder indexes the projects that build on it.

| Project | What it adds | Domain | Paper | Code |
| --- | --- | --- | --- | --- |
| [**OpenForge RL**](openforge-rl.md) | Trains agents inside their *real* deployment harnesses (ZeroClaw, OpenClaw, Codex) via a recording proxy, so no harness reimplementation is needed | CLI/tool agents · computer use · browser | [arXiv:2607.21557](https://arxiv.org/abs/2607.21557) | [MSR-Orchard/OpenForge-RL](https://github.com/MSR-Orchard/OpenForge-RL) |
| [**OpenWebRL**](openwebrl.md) | Online multi-turn RL on *live* websites — fault-tolerant browser environment, multimodal context management, trajectory-level success judging | Browser use | [arXiv:2606.02031](https://arxiv.org/abs/2606.02031) | [OpenWebRL/OpenWebRL](https://github.com/OpenWebRL/OpenWebRL) |
| **Orchard-SWE** | Credit-assignment SFT · Balanced Adaptive Rollout · on-policy distillation · rubric-based process reward — **73.0%** SWE-bench Verified | Software engineering | [arXiv:2605.15040](https://arxiv.org/abs/2605.15040) | [`examples/orchard_swe/`](https://github.com/MSR-Orchard/slime/tree/main/examples/orchard_swe) |
| **Orchard-GUI** | Distillation, then online RL on live websites — **68.4%** average across three browser benchmarks | Browser use | [arXiv:2605.15040](https://arxiv.org/abs/2605.15040) | [`examples/orchard_gui/`](https://github.com/MSR-Orchard/slime/tree/main/examples/orchard_gui) |
| **Orchard-Claw** | Opus-synthesized tasks, trained across two harnesses — **59.6%** pass@3, **73.9%** under ZeroClaw | CLI/tool agents | [arXiv:2605.15040](https://arxiv.org/abs/2605.15040) | — |

The bottom three are the recipes described in the Orchard paper itself; see
[Recipes](../README.md#recipes) in the root README. The top two are follow-on
work with their own papers and repositories.

## Adding your project

The [REST API](../orchard_env/docs/api.md) is the contract and the Python SDK is
a thin client over it, so a project in any language can depend on the same
substrate. To list yours here:

1. Add a row to the table above.
2. Add a short page in this folder — copy the shape of
   [`openforge-rl.md`](openforge-rl.md): what it adds on top of Orchard, which
   Orchard Env features it relies on, headline results, and links to the paper
   and the code.
3. Open a PR.

Please keep pages short and factual. This index is a map to other repositories,
not a mirror of their documentation.
