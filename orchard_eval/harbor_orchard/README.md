# harbor-orchard

Run **Harbor** tasks — Terminal-Bench and anything else in the Harbor format —
on **Orchard Env** sandboxes.

Harbor already owns everything about a benchmark run that is easy to get subtly
wrong: dataset resolution from the Hub, the oracle agent, uploading `/tests`,
parsing reward files, separate verifier environments, artifact transfer between
them. What it does *not* own is where the container comes from. That is a
pluggable `BaseEnvironment`, and this package is one — no fork, no patch.

```bash
harbor run -d terminal-bench/terminal-bench@4.0.0 \
    --env harbor_orchard:OrchardEnvironment \
    --agent oracle -n 32
```

## The two problems this solves

Orchard's sandbox service is a Kubernetes pod service. It cannot build images
from a Dockerfile, and it cannot bind-mount host directories. Harbor tasks
assume both. The gap turns out to be narrower than it looks:

| Harbor expects | Orchard provides | What this package does |
| --- | --- | --- |
| `docker build` from `environment/Dockerfile` | A pod from a registry image | Starts the pod from the Dockerfile's `FROM`, then **replays every remaining instruction as a command** |
| Bind-mounted `/logs` | Nothing | Declares `capabilities.mounted = False`, which makes Harbor *download* `/logs` — a path the framework already has |
| Multi-stage `COPY --from=builder` | One pod | Builds earlier stages in throwaway pods and tars the result across |
| A second verifier container | A second pod | Falls out for free: Harbor asks for another environment whose build context is `tests/` |

The mount problem is therefore a *declaration*, not an implementation. The
Dockerfile problem is the real work, and it is what most of this package is.

## The translation layer

`environment/Dockerfile` → `harbor_orchard.dockerfile` (parse) →
`harbor_orchard.plan` (interpret) → `harbor_orchard.builder` (execute).

A `docker build` carries state between layers that a sandbox `exec` does not:
each `exec` is a fresh process that remembers no working directory, no user, no
environment. So `ENV`, `WORKDIR`, `USER`, `ARG` and `SHELL` are *interpreted* at
translation time and attached to every emitted command, while `RUN`, `COPY` and
`ADD` become operations. The final state becomes the sandbox's defaults, because
the agent and the verifier have to run in the environment the build left behind.

Three details carry most of the risk, and each has tests pinning it:

**`ENV PATH="/opt/x:${PATH}"`** — the most common Dockerfile idiom there is.
`${PATH}` refers to the *base image's* PATH, so the base environment is read out
of the pod with `env -0` before translation begins. Expanding it against an
empty scope would replace the system PATH with a single directory and break
every command after that line.

**`RUN` is not expanded here.** Docker expands `$VAR` in `COPY`/`WORKDIR`/`ENV`
at build time and leaves `RUN` to the shell at run time. Both behaviours are
reproduced; collapsing them resolves variables against the wrong scope.

**`COPY dir dest` copies the directory's *contents*.** This surprises people and
Terminal-Bench depends on it (`COPY data /app/data/`). The placement rules live
in one POSIX shell routine in `harbor_orchard/shell.py` rather than being
re-derived per call site, and directories are staged and applied by that single
routine whether they came from the build context, another stage, or a URL.

Transfers are chunked and tar-based. The Orchard file API carries a whole file
as base64 in one JSON request, and Terminal-Bench ships reference genomes; a
single-shot transfer would build a half-gigabyte request body. Directories move
as one tar stream because per-file transfer silently drops executable bits and
symlinks — which would land a task's `solve.sh` non-executable.

One case does not fit that shape. `COPY --from` accepts an *image*, and those
images are routinely distroless — `COPY --from=ghcr.io/astral-sh/uv:0.8.14 /uv
/bin/` appears in Terminal-Bench, and that image holds two binaries and nothing
else: no shell, no `tar`, no `cp`. When the source pod has no shell, the files
are pulled through the in-pod agent's HTTP API, which is a Python service and
needs none. Permission bits are the casualty there — the agent reports name,
type and size but not mode — so everything fetched that way is made executable,
which is right for the tool images this path exists for.

## What it refuses, and why that is the point

Harbor validates a task against the provider's declared capabilities *before*
creating anything. Declaring honestly means an unsupported task fails
immediately with a clear reason, instead of running in an environment that does
not match what the task author specified and scoring a model on it.

| Declared `False` | Effect |
| --- | --- |
| `docker_compose` | Multi-container tasks are rejected at load time |
| `gpus` | GPU tasks are rejected |
| `windows` | Windows-container tasks are rejected |

## Agents that do not reinstall what the pod has

The environment is most of this package, but not all of it.
[`harbor_orchard/agents.py`](harbor_orchard/agents.py) ships two thin subclasses
of Harbor's own installed agents, and `orchard-eval harbor` selects them for the
short names `pi` and `mini-swe-agent`.

Harbor's installed agents fetch their CLI at trial time, from inside the task's
image. Orchard pods do not need that — the orchestrator mounts a prebuilt,
self-contained payload at `/opt/sandbox-tools` and shims it into
`/usr/local/bin` — and on SWE-bench Pro the fetch is where the trials went:

| agent | what Harbor does | what it costs |
| --- | --- | --- |
| `pi` | nvm + `npm install -g` | 88 of 731 trials. nodejs.org publishes no musl build, so nvm 404s on every Alpine image and `pi` exits 127 — reported as `NetworkConnectionError` |
| `mini-swe-agent` | `uv tool install` | 88 of 731 trials. 69 fell back to a Rust sdist with no toolchain to build it; 16 installed against a system Python 3.10 and died importing `typing.NotRequired` |
| `codex` | *skips its install when codex is on PATH* | nothing — and its payload is the static-PIE musl build |

Because both failures happen inside `install()`, the trial is over before
`run()` would have invoked the CLI: a working payload is never reached. So the
subclasses decide up front, in this order:

1. **the payload CLI runs, and no version was pinned** — skip Harbor's install;
2. **it does not run and the image is musl** — raise, naming that as the cause,
   instead of letting nvm or uv fail for a reason of their own;
3. **otherwise** — `super().install()`, so an image with no payload mounted, or
   a run that pinned a version, behaves exactly as it does upstream.

Step 1 *runs* the CLI rather than looking for it with `command -v`. A glibc
payload on a musl host resolves on PATH and then fails to `execve`, so presence
and usability are different questions, and only the second one matters.

Adopting the payload is also the point at which the image is known to be one the
payload had to be carried into, so it is where the remaining POSIX gaps get
filled. There is one so far: Harbor ends pi's run with `| stdbuf -oL tee`, and
`stdbuf` is GNU coreutils, which busybox does not carry. That stage exits 127,
`set -o pipefail` propagates it, and the trial is recorded as an agent failure —
this time *after* the pod and the model have been paid for. The shim drops the
buffering flags and runs the command; it is written only when the image has no
real `stdbuf`, so a glibc image is untouched.

`ORCHARD_HARBOR_STOCK_AGENTS=1` opts back out to Harbor's own classes, for
comparing against upstream behaviour.

One thing to know before comparing runs across this change: on a **glibc**
image, where Harbor's installer did work, it installed `@latest` on the day of
the run. The payload is whatever version the `sandbox-tools` image was built
with, which is pinned and reproducible but not the same thing. Passing
`--ak version=<x>` still routes through Harbor's installer, so an exact version
is always available when a comparison needs one.

## Stopping an agent Harbor has stopped waiting for

Harbor enforces its agent deadline with an `asyncio.wait_for` around the agent
coroutine. That cancels the *await* and nothing else: the CLI goes on running
inside the pod, and every call Harbor makes next — the agent-log sync, artifact
collection, the verifier — queues behind it.

Measured on a 731-trial SWE-bench Pro run (`n-concurrent 32`, agent timeout
3000s), by exit path:

| agent exit | trials | median gap to verifier start | gap > 1800s |
| --- | --- | --- | --- |
| clean | 209 | 1s | **0 (0%)** |
| `AgentTimeoutError` | 519 | 3s | **176 (34%)** |

The gap is the orphan: in the stalled trials whose logs survived, the last
model call is timestamped a median of **2687s after** the recorded
`agent_execution.finished_at`, and the verifier starts within ~3s of that last
call. It cost 174 GPU-hours — 28% of the run's whole agent budget — and aged
189 pods past `SANDBOX_TTL_HOURS` while they were blocked, so those trials were
never graded at all (`reward: null`, counted against the solve rate).

The fix is to bound the CLI *inside* the pod, a margin before Harbor's own
deadline:

* `environment.py` reads the task's `[agent] timeout_sec` from the `task.toml`
  beside its `environment/` directory, multiplies by
  `ORCHARD_HARBOR_AGENT_MULTIPLIER`, subtracts `ORCHARD_HARBOR_AGENT_MARGIN`,
  and exports the result as `ORCHARD_AGENT_DEADLINE` into every exec.
* `agents.py` publishes each payload CLI behind a shim that runs it under
  `timeout -k 10 "$ORCHARD_AGENT_DEADLINE"`. The wrapper goes around the CLI
  only, never the `| tee /logs/agent/...` Harbor builds around it, so when the
  deadline fires the CLI dies, its pipe closes, and `tee` drains and flushes
  normally. That is the whole point of stopping early: the transcript and the
  diff come back readable, where a rollout cut off at the exec deadline loses
  both.
* `timeout` exits 124, which Harbor raises as `NonZeroAgentExitCodeError`. The
  environment names the deadline in stderr and
  `orchard_evalkit.harbor_bridge` reads that back as `TIMEOUT`, so a budget
  that is too small is not filed under "the CLI is broken".
* An image with no `timeout`, or a CLI that outlives `SIGKILL`'s grace, falls
  through to a backstop: `exec` catches the `CancelledError` Harbor's
  `wait_for` raises and stops any process running from the payload directory.

This is the same fix as commit `9e39b7a` on the `orchard-eval run` path, which
did not touch this package.

## Network policy

`network_mode = "no-network"` used to be a rejection. It is now honoured, which
whole benchmarks depend on: every DeepSWE task is air-gapped, so declaring
`disable_internet = False` refused all 113 of them at load time.

Orchard fixes egress when a pod is created, but Harbor scopes network policy to
*phases*: the pod starts on the `[environment]` baseline so `agent.setup()` can
install an agent CLI, and Harbor then calls `set_network_policy()` to apply the
`[agent]` override before `agent.run()`. DeepSWE is written exactly that way —
no `[environment].network_mode`, so the baseline is `public`, with `no-network`
on `[agent]` and `[verifier]`. A provider that cannot switch is rejected at
trial init:

```
ValueError: [agent] agent phase network policy differs from the agent
environment baseline, but this environment cannot change network policy after start.
```

So the provider declares `dynamic_network_policy` and implements
`_apply_network_policy`, backed by `PUT /sandboxes/{id}/network` on the
orchestrator, which rewrites the pod's NetworkPolicy in place. That route has
carried two payload shapes, so `_switch_network` dispatches on what the SDK
exposes — `disable_network` / `enable_network` with per-port rules, or the older
`set_network` with bare CIDRs — rather than pinning one and requiring a flag day.
An orchestrator offering neither is named as such, because an un-isolated run
that still produces a score is worse than a stopped one.

An isolated pod still keeps an allowlist for the destinations an in-pod agent
legitimately needs — the pinned model endpoint, plus `ORCHARD_HARBOR_EGRESS_ALLOW`
— resolved to CIDRs. Everything else is denied, so the repository under test is
unreachable and the upstream fix cannot leak in.

The allowlist is per *port*: a destination given as a URL or `host:port` opens
only that port, and anything else opens `ORCHARD_HARBOR_EGRESS_PORTS` (80 and
443 by default).

There is no DNS exemption, and none can be expressed — the orchestrator requires
a CIDR per rule and rejects `0.0.0.0/0`. So an isolated pod cannot resolve
names, and is handed the endpoint's *address* instead: `_pin_endpoint` rewrites
the base URL once, before the allowlist is derived from it, so a round-robin
name cannot allowlist one address while the agent dials another. Set
`ORCHARD_HARBOR_RESOLVE_ENDPOINT=0` for a load-balanced name that must keep its
full address set, and allowlist the DNS server instead.

`network_allowlist` is declared **False**: Harbor's allowlist mode names *hosts*,
and a pod-level NetworkPolicy allows *addresses*. Rejecting such a task is
honest; enforcing a weaker rule than the author asked for is not. Neither
DeepSWE nor terminal-bench uses that mode.

| Variable | Meaning |
| --- | --- |
| `ORCHARD_HARBOR_MODEL_EGRESS=0` | Seal the pod completely. Correct for `oracle`, which never calls a model. |
| `ORCHARD_HARBOR_EGRESS_ALLOW` | Extra destinations, comma separated: hostnames, URLs or CIDRs. For a fleet behind a load balancer whose address set is wider than one DNS answer. |
| `ORCHARD_HARBOR_EGRESS_PORTS` | Ports opened for a destination that names none of its own. Default `80,443`. |
| `ORCHARD_HARBOR_RESOLVE_ENDPOINT=0` | Hand the agent the endpoint's hostname rather than its address. Needs the DNS server allowlisted. |
| `ORCHARD_HARBOR_ALLOW_INTERNET=1` | Ignore the task's air-gap and give the pod full egress. See below. |
| `ORCHARD_HARBOR_FORCE_ISOLATION=1` | The mirror of the above: ignore the task's *declared* access and isolate the pod anyway. For a benchmark that ships `allow_internet = true` on every task, like SWE-bench Pro. |

A host that cannot be resolved is logged and skipped rather than raised: one bad
name should not turn a 113-task sweep into a configuration error, but it *will*
make that agent fail, so the warning names it.

### Ignoring the air-gap

`ORCHARD_HARBOR_ALLOW_INTERNET=1` exists for bringing a benchmark up — a build
that needs packages, or an in-pod agent whose allowlist is not right yet. It
warns once per environment and the reason is worth restating: DeepSWE's images
clone the repository, reset to the base commit, then `git gc` the future away,
with the authors' own comment reading *"future commits/tags gc'd away so the
reference solution can't leak from history"*. Egress undoes that with one `git
clone`. Use it to test correctness, not to produce a number you will compare
with anything.

## Audit before you spend anything

`harbor-orchard audit` walks a dataset and reports what will run, offline, in
seconds, without creating a pod:

```bash
harbor-orchard audit /path/to/terminal-bench/tasks
harbor-orchard audit /path/to/terminal-bench/tasks --json > audit.json
```

In a `terminal-bench` checkout, `tasks/` is the 4.0 set and `archive/` holds the
2.x tasks. Datasets pulled from the Hub unpack under `~/.cache/harbor/`, so they
can be audited the same way.

```
  UNSUPPORTED  erp-procurement-planning       docker-compose: needs several networked containers
  UNSUPPORTED  jax-speedrun-gpu               gpus=1: sandboxes have no GPU support

  runnable       55/66 = 83.3%
  air-gapped      2/66 (run isolated; an in-pod agent needs ORCHARD_HARBOR_EGRESS_ALLOW or the pinned model endpoint)
    11  docker-compose
     4  gpus
```

Air-gapped is a count, not a rejection: those tasks run, with the pod isolated.
Pointed at DeepSWE it reads `air-gapped 113/113`, which is the number to check
before concluding that a sweep of zeros was the model's fault.

It exits non-zero only when a Dockerfile could not be *translated*, so it works
as a CI gate: point it at a new Terminal-Bench release and any Dockerfile
feature that release introduced shows up as a failure rather than as a mystery
mid-benchmark.

To see the actual commands for one task:

```bash
harbor-orchard translate /path/to/terminal-bench/tasks/hello-world --show-env
```

## Install

The provider is imported **by the `harbor` process**, so it has to be importable
from the same environment Harbor runs in. This is the one setup mistake that
produces a confusing error.

`orchard-env` and `harbor-orchard` are not published to PyPI — they live in this
repository — so they have to be installed by path. From the repository root:

```bash
python -m pip install harbor
python -m pip install -e orchard_env -e orchard_eval/harbor_orchard
```

`python -m pip` rather than `pip`: it guarantees the packages land in the
interpreter you are about to run, which is the whole point of the check below.

On a Debian or Ubuntu base image the first command may fail with
`uninstall-no-record-file` for `PyJWT` — apt installed it, so pip cannot remove
it. Install a newer one alongside rather than replacing it; `/usr/local`
precedes `/usr/lib/python3` on `sys.path`, so the new one wins and apt stays
consistent:

```bash
python -m pip install --ignore-installed PyJWT harbor
```

If you prefer uv's isolated tool environment, inject them as editable paths
(`--with` resolves from the registry and will fail on these names):

```bash
uv tool install harbor \
    --with-editable ./orchard_env \
    --with-editable ./orchard_eval/harbor_orchard
```

Either way, verify the provider is visible to Harbor before running a job — and
run the check from **outside** the repository:

```bash
cd /tmp && python -c "import harbor, harbor_orchard; print(harbor_orchard.__file__)"
```

The `cd` is not incidental. `python -c` puts the current directory first on
`sys.path`, and `orchard_eval/` — the directory the eval scripts run from —
contains this project's `harbor_orchard/` directory, which has no `__init__.py`.
With nothing installed, Python imports that as an empty namespace package rather
than failing, and the check reports a misleading
`module 'harbor_orchard' has no attribute 'OrchardEnvironment'`. A `__file__` of
`None` means the same thing. Console scripts (`harbor`, `harbor-orchard`) are
unaffected — they put their own `bin` directory on the path, not your cwd.

Auditing needs neither Harbor nor a cluster:

```bash
python -m pip install -e orchard_eval/harbor_orchard
```

## Configure

Only `SANDBOX_BASE_URL` is required. Everything else has a working default.

```bash
export SANDBOX_BASE_URL="http://your-orchestrator-host"
export SANDBOX_API_KEY="your-sandbox-key"
```

| Variable | Default | Notes |
| --- | --- | --- |
| `SANDBOX_BASE_URL` | — | Required. Checked at job preflight, not at trial 1 |
| `SANDBOX_API_KEY` | — | Sent as `X-API-Key` |
| `SANDBOX_PREFIX` | — | Prefix for sandbox ids |
| `ORCHARD_HARBOR_IMAGE_MIRROR` | `mirror.gcr.io` | Set empty to disable rewriting |
| `ORCHARD_HARBOR_IMAGE_REMAP` | DeepSWE's ECR repo → `wenlinyao/deep-swe` | `src=dst,…` repository substitutions. Set empty to disable |
| `ORCHARD_HARBOR_STEP_TIMEOUT` | `3600` | Seconds for one replayed instruction |
| `ORCHARD_HARBOR_EXEC_TIMEOUT` | `3600` | Seconds for an agent/verifier command. Must cover the task's own `[agent] timeout_sec` — DeepSWE allows 10800 |
| `ORCHARD_HARBOR_AGENT_MARGIN` | `120` | Seconds to stop the in-pod CLI *before* Harbor's agent deadline. `0` restores the old behaviour, where a timed-out agent keeps running |
| `ORCHARD_HARBOR_AGENT_MULTIPLIER` | `1.0` | Mirror of Harbor's `--agent-timeout-multiplier`, which this package cannot see. `orchard-eval harbor` exports it automatically; set it by hand only for a bare `harbor run` |
| `ORCHARD_HARBOR_LIVENESS_INTERVAL` | `120` | Seconds between checks that the pod a call is waiting on still exists. `0` disables, and a reaped pod then costs a whole `ORCHARD_HARBOR_EXEC_TIMEOUT` |
| `ORCHARD_HARBOR_CREATE_TIMEOUT` | `1800` | Seconds to wait for a pod |
| `ORCHARD_HARBOR_CHUNK_MB` | `16` | Transfer chunk size |
| `ORCHARD_HARBOR_DEFAULT_CPU` | `4` | For tasks declaring no `cpus` |
| `ORCHARD_HARBOR_DEFAULT_MEMORY` | `16Gi` | For tasks declaring no `memory_mb` |
| `MODEL_BASE_URL` | — | Endpoint the in-pod agent talks to |
| `MODEL_BASE_URL_REPLICAS` | `1` | Read the above as the first of N consecutive ports |
| `MODEL_BASE_URLS` | — | Comma-separated endpoints; wins over the two above |
| `MODEL_ROUTING` | `sticky` | `sticky` hashes the task name, `random` draws per trial, `session` names it in the path for `scripts/session_router.py` and closes it at teardown |
| `ORCHARD_ROUTER_CONTROL_URL` | — | Host to send that close to when it is not the one in `MODEL_BASE_URL` — e.g. `http://127.0.0.1:8100`. The endpoint is written for the sandboxes; the close is made from the orchestrator, which may not be able to reach the same tunnel |
| `MODEL_WIRE_API` | `responses` | Protocol the staged codex config declares |
| `MODEL_NAME` | — | Served model id, declared in pi's catalog as `orchard-model` |
| `API_KEY` | — | Credential the in-pod agent presents |
| `ORCHARD_HARBOR_STAGE_AGENT_CONFIG` | `true` | Write the agent provider config into the pod |
| `ORCHARD_HARBOR_PI_MAX_TOKENS` | `16384` | pi's `maxTokens`. Matches `configs/pi.yaml`'s `max_tokens` so the two paths stay comparable. Empty omits the field and lets pi choose. Raising it trades truncated rollouts for timed-out ones — see the note below |
| `ORCHARD_HARBOR_COMMIT_BEFORE_COLLECT` | `true` | Commit the worktree before a collect hook that diffs against `HEAD`. Without it, DeepSWE 1.1 grades an empty patch for any agent that did not commit — which is most of them |
| `ORCHARD_HARBOR_COMMIT_AUTHOR_NAME` | `orchard-eval` | Identity for that commit, passed with `git -c` so it cannot outrank a task's own `git config` |
| `ORCHARD_HARBOR_COMMIT_AUTHOR_EMAIL` | `orchard-eval@local` | As above |
| `ORCHARD_HARBOR_MODEL_EGRESS` | `true` | Keep the pinned endpoint reachable from an isolated pod. `0` seals it |

### Committed work, and why a rollout can score zero having done everything

DeepSWE 1.1 is the only benchmark here whose tasks declare a
`[[verifier.collect]]` hook. It reads `git diff <base_commit> HEAD`, and a
separate verifier container applies that patch to a pristine checkout — so a
rollout is graded on what `HEAD` points at, not on what is in the working tree.
The task's own `solution/solve.sh` says it outright: *"only committed work is
graded"*. Neither pi nor mini-swe-agent commits by default, and most rollouts
are cut off by the agent timeout before they would; on one 113-task run that
left 96 of 113 pi trials and 64 of 113 mini-swe-agent trials with a zero-byte
`model.patch`, none of which scored, while the trials that *did* produce a patch
solved at 40% and 48%.

`ORCHARD_HARBOR_COMMIT_BEFORE_COLLECT` closes that by committing the worktree
just before the first such hook runs — in `OrchardEnvironment.service_exec`,
which Harbor reaches for collect hooks and nothing else, and therefore on the
timeout path as well as the clean one. It is a no-op when the agent committed
for itself, and tasks with no collect hooks (SWE-bench Pro, Terminal-Bench 2.1,
both graded in the live container) never reach it.

Turning it off is a defensible choice, not a bug: committing is literally what
the benchmark's instruction asks the agent to do, so leaving it off measures
that too.

### Why `ORCHARD_HARBOR_PI_MAX_TOKENS` is not a free knob

pi ends the whole session on the first completion that comes back truncated,
rather than re-prompting, so this cap decides how many rollouts survive. Raising
it does not remove those rollouts — it converts them into timeouts. Measured on
SWE-bench Verified, same model, 16384 → 32768:

| | 16384 | 32768 |
| --- | --- | --- |
| ended on `stopReason=length` | 95/500 | 40/500 |
| reached the wall-clock | 29/500 | 146/500 |
| median rollout | 828s | 1090s |
| resolve rate | 70.0% | 68.8% |

At ~25 output tok/s a single 32768-token runaway `thinking` block costs ~22
minutes of a 30-minute budget. Treat this and the agent timeout as one knob.

Which is why `scripts/run_all_evals.sh` raises it to 32768 for DeepSWE 1.1 and
leaves every other stage at 16384: the same block costs ~22 minutes of a
135-minute budget there (5400s x 1.5). The cost of truncation on that benchmark
is measured — 84% of finished pi trials on the 2026-09-21 sweep ended on
`stopReason=length`, and they solved 6.9% of their tasks against 66.7% for the
ones that ended cleanly, because a session that ends mid-turn commits nothing
and DeepSWE grades the commit. `DEEPSWE_PI_MAX_TOKENS` overrides that stage.
| `ORCHARD_HARBOR_EGRESS_ALLOW` | — | Extra destinations for an isolated pod: hostnames, URLs or CIDRs |
| `ORCHARD_HARBOR_EGRESS_PORTS` | `80,443` | Ports opened for a destination that names none of its own |
| `ORCHARD_HARBOR_RESOLVE_ENDPOINT` | `true` | Give the agent the endpoint's address, so an isolated pod needs no DNS |
| `ORCHARD_HARBOR_ALLOW_INTERNET` | `false` | Ignore a task's `no-network`. Correctness testing only |
| `ORCHARD_HARBOR_FORCE_ISOLATION` | `false` | Isolate the pod whatever the task declares. Mutually exclusive with the above |
| `ORCHARD_HARBOR_PREFER_IPV4` | `true` | Make `localhost` resolve to 127.0.0.1 before `::1` inside the pod |
| `ORCHARD_HARBOR_REQUIRE_CLEAN_TREE` | `true` | Fail a trial whose repository is already modified when the agent starts |

`ORCHARD_HARBOR_REQUIRE_CLEAN_TREE` is the one knob here whose default costs
trials on purpose. A benchmark number is a solve rate at a fixed budget *from a
fixed starting point*, and on the 2026-09-24 SWE-bench Pro V2 sweep 96 of 642
trials did not have the starting point: their agent opened a repository in
which the task's own gold-patch targets were already written, because a
re-submitted `POST /exec` had left an earlier agent running in that pod and
started a second one on top of its finished work. Nothing failed, so the run
reported a score that mixed two protocols. With the guard on, those arrive as
`DirtyWorkingTreeError` before the model is paid for anything, and
`TASK_IDS=` in `scripts/run_swebench_pro_v2.sh` re-runs exactly that list.

Tracked files only, which is the same measure upstream's leak probe reports as
`MODIFIED`: a pristine V2 image legitimately carries untracked files. A check
that cannot run — no git, no checkout, an exec that failed — lets the trial
proceed and says so in the log, because it is not evidence of anything. Turn it
off for a dataset whose images ship modifications deliberately.

`ORCHARD_HARBOR_PREFER_IPV4` is environment parity, not a grading change. A
benchmark's own test suite routinely starts a server and then talks to it over
`http://localhost:<port>`; a server that binds the IPv4 wildcard — NodeBB logs
"listening on 0.0.0.0:4568" — is simply not there on `::1`. Upstream never sees
this, because Node asks `getaddrinfo` with `AI_ADDRCONFIG` and a single-stack
container has no global IPv6 address for glibc to keep `::1` on the strength of.
A **dual-stack Kubernetes pod does**, RFC 3484 sorts it first, and the
connection is refused before the server is ever consulted.

Both zeros on the 2026-09-24 SWE-bench Pro V2 oracle run were exactly that —
`connect ECONNREFUSED ::1:4568` against a live `0.0.0.0` listener — and all 23
passing trials had none. Setting this drops the `localhost` alias from the `::1`
line of `/etc/hosts` (what musl reads; Alpine images ignore `gai.conf`) and
appends `precedence ::ffff:0:0/96 100` to `/etc/gai.conf` (glibc's documented
prefer-IPv4 knob). It runs once per pod, is idempotent, and is best-effort: a
read-only `/etc` leaves the resolver exactly as the image shipped it. Turn it
off for a cluster whose pods are deliberately configured otherwise.

`ORCHARD_HARBOR_IMAGE_MIRROR` rewrites **Docker Hub references only**.
`python:3.11-slim` becomes `mirror.gcr.io/library/python:3.11-slim`, while
`mcr.microsoft.com/...` and `ghcr.io/...` are left alone — rewriting those onto
a Hub mirror produces a 404 several seconds into a trial.

`ORCHARD_HARBOR_IMAGE_REMAP` runs one step earlier and substitutes whole
repositories, keeping the tag, as `source=destination` pairs. It exists because
DeepSWE's 113 task images live in `public.ecr.aws`, whose anonymous pull quota
makes a pull outlive `ORCHARD_HARBOR_CREATE_TIMEOUT` — reported as
`EnvironmentStartTimeoutError`, which names nothing. The default table serves
them from the copy `orchard_eval/scripts/mirror_deep_swe_images.sh` pushes:

```
public.ecr.aws/d3j8x8q7/swe-bench-202605:<tag>
  -> mirror.gcr.io/wenlinyao/deep-swe:<tag>
```

```bash
export ORCHARD_HARBOR_IMAGE_REMAP="public.ecr.aws/d3j8x8q7/swe-bench-202605=yourorg/deep-swe"
export ORCHARD_HARBOR_IMAGE_REMAP=""   # empty: pull from the dataset's registry
```

Destinations are repositories, not references: a tag on one is rejected at
startup rather than concatenated with the source's. `harbor-orchard audit
<tasks> --json` reports the resolved `base_images`, so what a pod would pull can
be checked before one exists.

### A fleet of model servers

More than one `MODEL_*` endpoint makes the provider pin **each trial to one
server** for its whole agent loop, by overriding `OPENAI_BASE_URL` /
`OPENAI_API_BASE` in the pod on every `exec`. Harbor runs all its trials in one
process, so a process-wide environment variable cannot do this; the provider
can, because the agent runs in a pod it owns.

```bash
export MODEL_BASE_URL="http://<frps-public-ip>:30021/v1"   # ports 30021..30028
export MODEL_BASE_URL_REPLICAS=8
```

The assignment is a hash of the **task name**, so a retried trial returns to the
server that already holds its prefix rather than re-prefilling on a cold one.
The choice is logged as `pinned <task> to <url>` when the trial starts.

Setting `MODEL_BASE_URL` also stages `/var/tmp/orchard-codex/config.toml` into
the pod and exports `CODEX_HOME` to match. That file is not optional: codex
honours `OPENAI_BASE_URL` only for its *built-in* provider, which talks to
api.openai.com whatever the variable says, so without a declared provider every
trial dies on turn 1 with a non-zero exit that reads as an agent failure. The
same file switches off the tool groups codex sends as Responses API `namespace`
entries, which vLLM and SGLang reject outright.
`ORCHARD_HARBOR_STAGE_AGENT_CONFIG=false` turns all of it off.

This only reaches agents that run *inside* the environment. An agent whose model
loop runs in the harbor process instead reads its own litellm environment and is
not pinned.

Every exec also exports `LITELLM_LOCAL_MODEL_COST_MAP=True`. LiteLLM otherwise
fetches its model-cost map from raw.githubusercontent.com on first import, and
an isolated pod reaches only the pinned endpoint — DNS included — so the lookup
blocks until it gives up. On DeepSWE 1.1 that cost 38–57 minutes of the agent's
budget before mini-swe-agent's first model call, in 11 of the 13 rollouts that
survived to be read, and bought nothing: the fallback is the table bundled with
the package, which an air-gapped run was always going to use. pi has its own
switch, `PI_OFFLINE`, set alongside it.

## Reading the progress bar

Harbor's live bar carries exactly one number, and it chooses it by sorting the
verifier's reward keys and taking the first one:

```
 98/113 Solved: 0.214 (21/98) ━━━━━━━━━━━━━━━━╺━━━━━ 9:37:14 0:03:39
```

On a verifier that reports a single `reward` — Terminal-Bench, SWE-bench Pro —
that first key *is* the score. On DeepSWE it is not. Its verifier reports `f2p`,
`p2p`, `partial` and `reward`, `f2p` sorts first, and so the bar read
`F2P: 0.710` through a run that was scoring 0.21: the mean share of
fail-to-pass tests left green, which is partial credit the benchmark does not
award.

`harbor_orchard.progress` puts the graded score there instead — `reward`, which
is 1 only when every F2P test passes and no P2P test regresses — counted over
the trials that have finished, so the bar predicts the `solve_rate` in
`summary.json` instead of contradicting it. It installs itself when Harbor loads
the provider, changes nothing Harbor records or prints in its final table, and
hands the line back to Harbor untouched for any reward shape it cannot read.

## Sanity check

Every Harbor task ships `solution/solve.sh`, and the `oracle` agent replays it.
A dataset's oracle score is the ceiling on every model score you measure against
it afterwards, so run it first:

```bash
cd orchard_eval
./scripts/harbor_oracle_check.sh          # Terminal-Bench 2.1, threshold 0.98
```

Bring the provider up on **2.1** and measure models on **4.0**. Almost every 2.1
task is a single container with a shared verifier, so a failure there is a bug
in the translation layer rather than a feature the provider declined. The flip
side is that 2.1 uses a separate verifier environment in 2 of 90 tasks while 4.0
uses one in all 66 — so passing on 2.1 does *not* establish that the second-pod
path works. Cover it once 2.1 is green:

```bash
ORACLE_THRESHOLD=0.80 ./scripts/harbor_oracle_check.sh \
    terminal-bench/terminal-bench@4.0.0 16
```

Failures are grouped by cause, which is what makes a shortfall actionable:
`BUILD_FAILED` is a bug here, `REWARD_ZERO` means the solution ran and the
verifier disagreed, and `UNSUPPORTED_*` is a task this provider correctly
declined.

When a trial fails, the artifacts under the job directory say which step broke —
see [Debugging a failed trial](../README.md#debugging-a-failed-trial).

## Test

```bash
python -m pip install -e "orchard_eval/harbor_orchard[dev]"
python -m pytest orchard_eval/harbor_orchard/tests
```

The tests are offline and cover the parser, the translator, and mirror
rewriting. Every case corresponds to a construct that appears in a real
Terminal-Bench Dockerfile, and each one is a bug that would *not* raise — it
would produce a plausible command that does the wrong thing.
