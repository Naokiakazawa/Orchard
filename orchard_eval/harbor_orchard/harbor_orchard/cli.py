"""``harbor-orchard`` — inspect and audit the Dockerfile translation layer.

Two commands, both offline and both cheap:

``translate``
    Show what one Dockerfile becomes. Use it when a single task misbehaves and
    you want to see the exact commands the sandbox will run.

``audit``
    Walk a whole dataset and report which tasks this provider can run and why
    the rest cannot. Run it *before* a benchmark, not after: it turns "47 of 66
    tasks failed" into a list of named, categorised reasons in a few seconds,
    without creating a single pod.

Neither command talks to a cluster, so ``audit`` is also the regression test for
the translator — point it at a new terminal-bench release and any Dockerfile
feature that release introduced shows up as an unsupported entry.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from harbor_orchard.dockerfile import DockerfileError, parse_file
from harbor_orchard.images import rewrite as rewrite_image
from harbor_orchard.plan import CopyOp, RunOp, WriteFileOp, translate_stage

#: A plausible base-image environment. Translation needs *some* environment to
#: expand ``$PATH`` against; at audit time the real one is unknown because no
#: pod exists, and the difference cannot change whether a Dockerfile parses.
AUDIT_BASE_ENV = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "HOME": "/root",
    "HOSTNAME": "sandbox",
    "TERM": "xterm",
}


@dataclass
class TaskReport:
    name: str
    path: Path
    supported: bool = True
    reasons: list[str] = field(default_factory=list)
    base_images: list[str] = field(default_factory=list)
    steps: int = 0
    stages: int = 0
    #: Reported, not rejected: the provider isolates these pods rather than
    #: refusing them, and a whole benchmark can be air-gapped (DeepSWE is).
    isolated: bool = False

    def reject(self, reason: str) -> None:
        self.supported = False
        self.reasons.append(reason)


def audit_task(task_dir: Path) -> TaskReport:
    """Check every Dockerfile a task would need, agent and verifier alike."""
    report = TaskReport(name=task_dir.name, path=task_dir)

    if (task_dir / "environment" / "docker-compose.yaml").exists() or (
        task_dir / "environment" / "docker-compose.yml"
    ).exists():
        report.reject("docker-compose: needs several networked containers")

    config = _read_task_config(task_dir)
    if config.get("gpus"):
        report.reject(f"gpus={config['gpus']}: sandboxes have no GPU support")
    if config.get("os") not in (None, "linux"):
        report.reject(f"os={config['os']}: only Linux containers are supported")
    report.isolated = bool(config.get("no_network"))

    for role, dockerfile_path in _dockerfiles(task_dir):
        try:
            _audit_dockerfile(dockerfile_path, report)
        except DockerfileError as exc:
            report.reject(f"{role}: {exc}")
        except Exception as exc:  # noqa: BLE001 - an audit must never crash a sweep
            report.reject(f"{role}: unexpected {type(exc).__name__}: {exc}")

    return report


def _dockerfiles(task_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for role, relative in (("environment", "environment"), ("verifier", "tests")):
        candidate = task_dir / relative / "Dockerfile"
        if candidate.exists():
            found.append((role, candidate))
    return found


def _audit_dockerfile(path: Path, report: TaskReport) -> None:
    dockerfile = parse_file(path)
    report.stages += len(dockerfile.stages)
    context = path.parent
    for stage in dockerfile.stages:
        plan = translate_stage(
            dockerfile, stage, context_dir=context, base_env=dict(AUDIT_BASE_ENV)
        )
        report.steps += len(plan.operations)
        if stage is dockerfile.final_stage:
            report.base_images.append(rewrite_image(plan.base_image))


def _read_task_config(task_dir: Path) -> dict:
    """Extract the few ``task.toml`` fields that decide supportability."""
    config_path = task_dir / "task.toml"
    if not config_path.exists():
        return {}
    try:
        import tomllib

        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - Harbor validates this properly later
        return {}

    environment = data.get("environment", {}) or {}
    return {
        "gpus": environment.get("gpus") or None,
        "os": environment.get("os"),
        "no_network": _no_network(data),
    }


def _no_network(data: dict) -> bool:
    """Whether any phase of the task is air-gapped.

    Schema 1.3 moved network policy out of ``[environment]``: a task states it
    once under ``[agent]`` and ``[verifier]``, and reading only the old place
    reports a fully isolated dataset as an unrestricted one. Every DeepSWE task
    is written that way.
    """
    scopes = [
        data.get("environment") or {},
        data.get("agent") or {},
        data.get("verifier") or {},
        ((data.get("verifier") or {}).get("environment")) or {},
    ]
    for scope in scopes:
        if scope.get("network_mode") == "no-network":
            return True
        if scope.get("allow_internet") is False:
            return True
    return False


def _cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.tasks_dir)
    task_dirs = sorted(
        path.parent for path in root.glob("*/task.toml")
    ) or sorted(path for path in root.iterdir() if path.is_dir())

    reports = [audit_task(task_dir) for task_dir in task_dirs]
    supported = [report for report in reports if report.supported]
    rejected = [report for report in reports if not report.supported]

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "task": report.name,
                        "supported": report.supported,
                        "reasons": report.reasons,
                        "steps": report.steps,
                        "stages": report.stages,
                        "base_images": report.base_images,
                        "isolated": report.isolated,
                    }
                    for report in reports
                ],
                indent=2,
            )
        )
    else:
        for report in rejected:
            for reason in report.reasons:
                print(f"  UNSUPPORTED  {report.name:<34} {reason}")
        print()
        total = len(reports) or 1
        print(f"  runnable       {len(supported)}/{len(reports)} = {len(supported) / total:.1%}")
        isolated = [report for report in supported if report.isolated]
        if isolated:
            print(
                f"  air-gapped     {len(isolated)}/{len(reports)} "
                "(run isolated; an in-pod agent needs ORCHARD_HARBOR_EGRESS_ALLOW"
                " or the pinned model endpoint)"
            )
        categories = Counter(
            reason.split(":", 1)[0] for report in rejected for reason in report.reasons
        )
        for category, count in categories.most_common():
            print(f"  {count:>4}  {category}")

    # A dataset with unsupported tasks is expected, not an error; only a
    # translation *failure* should fail the command, so that this can gate CI.
    broke = [
        report
        for report in rejected
        if any(reason.startswith(("environment:", "verifier:")) for reason in report.reasons)
    ]
    if broke:
        print(f"\n  {len(broke)} task(s) could not be translated at all.", file=sys.stderr)
        return 1
    return 0


def _cmd_translate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if path.is_dir():
        path = path / "environment" / "Dockerfile"
    dockerfile = parse_file(path)
    context = path.parent

    for stage in dockerfile.stages:
        label = stage.name or f"stage {stage.index}"
        print(f"# ===== {label}: FROM {rewrite_image(stage.base)}")
        plan = translate_stage(
            dockerfile, stage, context_dir=context, base_env=dict(AUDIT_BASE_ENV)
        )
        for operation in plan.operations:
            print(f"\n# {operation.source}")
            if isinstance(operation, RunOp):
                print(f"#   cwd={operation.cwd} user={operation.user or 'root'}")
                print(operation.command())
            elif isinstance(operation, CopyOp):
                origin = f"--from={operation.from_stage} " if operation.from_stage else ""
                names = " ".join(
                    source.name or source.remote_path or source.url or "?"
                    for source in operation.sources
                )
                print(f"# COPY {origin}{names} -> {operation.dest}")
            elif isinstance(operation, WriteFileOp):
                print(f"# WRITE {len(operation.content)} bytes -> {operation.dest}")
        print(f"\n# final: workdir={plan.state.workdir} user={plan.state.user or 'root'}")
        if args.show_env:
            for key, value in sorted(plan.state.env.items()):
                print(f"#   {key}={value}")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="harbor-orchard",
        description="Inspect the Dockerfile-to-shell translation used by the "
        "Orchard environment provider.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit",
        help="Report which tasks in a dataset directory this provider can run",
    )
    audit.add_argument("tasks_dir", help="Directory of Harbor task directories")
    audit.add_argument("--json", action="store_true", help="Emit a machine-readable report")
    audit.set_defaults(func=_cmd_audit)

    translate = subparsers.add_parser(
        "translate", help="Print the commands one Dockerfile becomes"
    )
    translate.add_argument("path", help="A task directory or a Dockerfile")
    translate.add_argument(
        "--show-env", action="store_true", help="Also print the resulting environment"
    )
    translate.set_defaults(func=_cmd_translate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
