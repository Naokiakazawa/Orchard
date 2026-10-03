# Python SDK reference

Import everything from the top-level package:

```python
from orchard_env import (
    SandboxClient,
    AsyncSandboxClient,
    SandboxInstance,
    AsyncSandboxInstance,
    JobResult,
)
```

See the [README](../README.md) for installation and quickstart examples.

## `SandboxClient` / `AsyncSandboxClient`

```python
SandboxClient(
    base_url: Optional[str] = None,  # falls back to SANDBOX_BASE_URL
    timeout: int = 1200,             # request timeout in seconds
    auto_cleanup: bool = True,       # delete tracked sandboxes on exit
    api_key: Optional[str] = None,   # falls back to SANDBOX_API_KEY
    prefix: Optional[str] = None,    # falls back to SANDBOX_PREFIX
)
```

| Method | Description |
| --- | --- |
| `create_sandbox(image, ...)` | Create a new sandbox container |
| `get_sandbox(sandbox_id)` | Get an existing sandbox by ID |
| `delete_sandbox(sandbox_id)` | Delete a sandbox |
| `cleanup_all()` | Delete every sandbox created by this client |

Both clients are context managers, and that is the recommended way to use them —
leaving the block guarantees cleanup.

### Sandbox creation options

```python
sandbox = client.create_sandbox(
    image="python:3.11-slim",    # container image
    block_network=True,          # block outbound network (default: True)
    sandbox_id=None,             # custom ID (auto-generated when None)
    cpu="4",                     # CPU cores, e.g. "4" or "2000m"
    memory="16Gi",               # memory limit
    timeout=3600,                # seconds to wait for readiness
    wait_ready=True,
    poll_interval=1.0,
)
```

Sandboxes have **no outbound network access by default**. Anything that reaches
the internet — `pip install`, `git clone`, or an agent harness calling a model
API — needs `block_network=False`.

## `SandboxInstance` / `AsyncSandboxInstance`

| Method | Description |
| --- | --- |
| `exec(command, timeout, cwd, env)` | Run a command in the sandbox |
| `apply_patch(patch, timeout)` | Apply a git patch |
| `upload_file(local_path, remote_path)` | Upload a file |
| `upload_content(content, remote_path)` | Upload bytes/str directly |
| `download_file(remote_path, local_path)` | Download a file |
| `download_content(remote_path)` | Download file content as bytes |
| `list_files(remote_path)` | List a directory |
| `get_job(job_id)` | Fetch job status and results |
| `get_network_config()` | Current egress mode and allowlist |
| `disable_network(allowlist)` | Restrict egress to CIDR/port rules |
| `enable_network()` | Restore unrestricted egress |
| `delete()` | Delete the sandbox |

`exec()` also accepts `login_shell=True` (runs under `bash -lc` instead of
`bash -c`) and `pty=True` for an interactive session backed by
`ContainerProcess` / `AsyncContainerProcess`.

### Changing egress on a running sandbox

Egress is otherwise fixed when the pod is created, which leaves no way to
install software that needs the network and then hand an isolated environment to
whatever runs next. Rewriting the per-sandbox NetworkPolicy does, because the
policy is evaluated per packet rather than at admission.

```python
with client.create_sandbox("python:3.11-slim", block_network=False) as sandbox:
    sandbox.disable_network(
        [
            {
                "cidr": "203.0.113.10/32",
                "protocol": "TCP",
                "port_start": 30000,
                "port_end": 31000,
            }
        ]
    )
    sandbox.get_network_config()
    sandbox.enable_network()
```

The async client exposes the same three methods with `await`.

A rule needs a `cidr` and a `port_start`; `protocol` defaults to `TCP` and
`port_end` to `port_start`. A bare address is normalized to a host prefix
(`/32`, or `/128` for IPv6) and `0.0.0.0/0` is rejected, so "allow everything"
has to be said as `enable_network()` rather than smuggled in as a rule.

Two limits worth knowing before relying on this:

- `disable_network([])` blocks all *new* egress connections. TCP connections
  already established are not guaranteed to be terminated.
- Allowlists are addresses, not names. There is no DNS-name allowlist, and no
  implicit exemption for port 53 — an isolated pod that is given a hostname
  cannot resolve it, so resolve to an address before handing one over.

## `JobResult`

```python
result = sandbox.exec("ls -la")

result.job_id       # unique job identifier
result.status       # "queued" | "running" | "succeeded" | "failed"
result.stdout
result.stderr
result.exit_code
result.succeeded    # True if exit_code == 0
result.failed       # True if status == "failed"
result.is_complete  # True once the job has finished
```

> A `"running"` status returned from `exec(wait=True)` means the *server-side*
> wait timed out, not that the job finished. The client transparently falls back
> to polling `GET /jobs/{id}` in that case.

## Auto cleanup

The client tracks every sandbox it creates and cleans them up on:

1. **Context manager exit** — leaving a `with` block deletes the sandbox
2. **Program exit** — remaining sandboxes are deleted via an `atexit` handler
3. **Signals** — cleanup also runs on `SIGINT` (Ctrl+C) and `SIGTERM`

Disable it with `SandboxClient(auto_cleanup=False)`.

## Error handling and retries

```python
import requests

from orchard_env import SandboxClient

try:
    with SandboxClient() as client:
        with client.create_sandbox("python:3.11-slim") as sandbox:
            result = sandbox.exec("exit 1")
            if not result.succeeded:
                print(f"Command failed: {result.stderr}")
except TimeoutError as e:
    print(f"Sandbox creation timed out: {e}")
except requests.exceptions.HTTPError as e:
    print(f"API error: {e}")
```

Transient failures (connection errors, timeouts, chunked-encoding errors, HTTP
503) are retried automatically: 3 attempts with exponential backoff and jitter
(1s, 2s, 4s).

The underlying HTTP endpoints are documented in [api.md](api.md).
