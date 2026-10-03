#!/usr/bin/env python3
"""Probe every SWE-bench Pro V1 image for git-history and hidden-file leaks.

V1 is ``scale-ai/swe-bench-pro@2`` — the 731 tasks the ``swebench-pro-harbor``
stages of run_all_evals.sh run. Its images are Scale's ``jefzda/sweap-images``
full clones, reset to the base commit by each task's Dockerfile (``git reset
--hard <base>``), and the fixing commit's SHA is in every task name. This
measures, image by image, whether the answer can be read out of the sandbox the
agent works in.

One thing is certain before running it: V1's verifier restores the hidden tests
with ``git checkout <fix_sha> -- <files>`` (all 731 tasks), so V1's grading
*needs* the fix commit inside the image. Stripping history from a V1 image
without also changing its verifier breaks grading — which is why upstream's
fix (scaleapi/SWE-bench_Pro-os#94) rewrites the harness too, and why V2 exists.

The probe is a Harbor task whose ``solve.sh`` reports on the sandbox instead of
solving it; run with the oracle agent, the report lands in each trial's
``agent/oracle.txt``, written during the agent phase with the agent's network
policy and view of the filesystem. The report is upstream's own leak-probe
template (``v2/tooling/probe/leakprobe_solve.sh.tmpl`` in SWE-bench_Pro-os, the
same one scripts/swebench_pro_v2_leakprobe.py renders) plus the checks below,
so a V1 and a V2 result are directly comparable.

Three steps:

    python scripts/swebench_pro_v1_leakprobe.py make --out /tmp/sbp-v1-leakprobe
    python scripts/swebench_pro_v1_leakprobe.py run  --tasks /tmp/sbp-v1-leakprobe/tasks \\
        --job-name smoke11 --sample 11
    python scripts/swebench_pro_v1_leakprobe.py report results/harbor/swebench-pro-v1-leakprobe/smoke11

``make`` reads Harbor's local cache of the dataset (``--tasks-dir``; any earlier
``swebench-pro-harbor`` run fills it). ``run`` needs SANDBOX_BASE_URL (and
SANDBOX_API_KEY) like every other Harbor path. V1 tasks declare
``allow_internet = true``, so the network checks report open unless ``run
--isolate`` air-gaps the agent phase, as ``PRO_ALLOW_INTERNET=0`` does in
run_all_evals.sh. The history checks do not depend on it.

Checks, all inside the agent-phase sandbox. A trial FAILS on any of:

  history — can the answer be read out of git?
    FIXSHA_OBJ != absent        the fixing commit (SHA in the task name) exists
    REFLOG, STASH > 0           reflog / stash entries survived
    UNREACHABLE_COMMITS > 0     commits reachable from nothing (``git fsck``)
    DANGLING_COMMITS > 0        same, ignoring reflogs
    ALL_OBJ_COMMITS != HEAD_COMMITS
                                some ref points at history HEAD does not have
    ODB_COMMITS_NOT_IN_HEAD > 0 commit objects anywhere in the object store that
                                HEAD cannot reach
    FIXED_BLOBS_IN_ODB > 0      a file's *post-fix* content is stored as a blob
                                HEAD's history does not contain — the answer,
                                recoverable without any commit at all
  workspace
    HIDDEN_TESTS_NEW_PRESENT    a test file the verifier checks out of the fix
                                commit, and HEAD does not have, is already on disk
    TESTS_DIR != 0              /tests is visible to the agent
    GOLD_ALREADY_APPLIED = 1    the working tree already contains the fix
  network
    NET_<host> = open           the agent phase reached a code host or index

Informational, never a failure: REFS, PACKED_REFS, HOOKS, ORIG_FETCH_HEAD,
FIX_REFS (the refs containing the fix commit — how ``git log --all`` finds it),
FIXED_BLOBS_IN_HEAD_HISTORY (post-fix content HEAD's own past already has: the
fix restores an old version, which is fair game), OTHER_GIT_DIRS and
SELF_NAMED_PATHS (paths outside the repo named after the project; a *newer*
installed release would be a leak no history check can see).

The verifier is replaced with one that writes reward 1 immediately: the report
is the output, and V1's real verifier would need the network and an hour.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
#: Where Harbor unpacks ``-d scale-ai/swe-bench-pro@2``: one dir per instance
#: holding a content-digest subdir.
DEFAULT_TASKS_DIR = Path("~/.cache/harbor/tasks/packages/scale-ai")
#: Upstream's probe template lives in the SWE-bench_Pro-os checkout that
#: scripts/fetch_swebench_pro_v2.sh maintains.
DEFAULT_TEMPLATE = (
    SCRIPT_DIR.parent
    / "third_party"
    / "SWE-bench_Pro-os"
    / "v2"
    / "tooling"
    / "probe"
    / "leakprobe_solve.sh.tmpl"
)
DEFAULT_JOBS_DIR = Path("results/harbor/swebench-pro-v1-leakprobe")

END_MARKER = 'echo "=== END LEAKPROBE ==="'
GOLD_START = "cat > solution_patch.diff << '__SOLUTION__'"

#: Hosts an agent would use to fetch the fix or a newer release of the project.
NET_HOSTS = [
    "github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
    "gitlab.com",
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "proxy.golang.org",
]

#: Before upstream's END marker, i.e. before the gold patch is applied. Plain
#: bash plus coreutils/busybox and git: some images have nothing else.
PRE_APPLY = r"""
# --- orchard_eval additions (scripts/swebench_pro_v1_leakprobe.py) ---
lp_repo=$(pwd)
lp_head_ct=$(git log -1 --format=%ct HEAD 2>/dev/null || echo 0)
git rev-list HEAD 2>/dev/null | sort > /tmp/.lp_reach
git cat-file --batch-all-objects --batch-check='%(objectname) %(objecttype)' 2>/dev/null \
    | awk '$2=="commit"{print $1}' | sort > /tmp/.lp_odb
comm -23 /tmp/.lp_odb /tmp/.lp_reach > /tmp/.lp_extra
echo "ODB_COMMITS=$(wc -l < /tmp/.lp_odb | tr -d ' ')"
echo "ODB_COMMITS_NOT_IN_HEAD=$(wc -l < /tmp/.lp_extra | tr -d ' ')"
if [ -s /tmp/.lp_extra ]; then
    echo "ODB_EXTRA_NEWER_THAN_HEAD=$(git log --no-walk=unsorted --stdin --format=%ct < /tmp/.lp_extra 2>/dev/null | awk -v h="$lp_head_ct" '$1>h' | wc -l | tr -d ' ')"
fi
echo "ALTERNATES=$(cat .git/objects/info/alternates 2>/dev/null | wc -l | tr -d ' ')"
if git cat-file -e __FIX_SHA__ 2>/dev/null; then
    echo "FIX_REFS=$(git for-each-ref --contains __FIX_SHA__ --format='%(refname)' 2>/dev/null | head -5 | tr '\n' ' ')"
fi
if git apply --check -R /solution/gold_patch.diff >/dev/null 2>&1; then
    echo "GOLD_ALREADY_APPLIED=1"
else
    echo "GOLD_ALREADY_APPLIED=0"
fi
# V1 ships no test patch; its verifier runs `git checkout <fix> -- <files>`. A
# listed file HEAD's tree lacks is one the fix creates, so finding it on disk
# means the hidden test is already in the workspace.
lp_new=0; lp_new_present=0; lp_new_files=""
for lp_f in __TEST_FILES__; do
    git cat-file -e "HEAD:$lp_f" 2>/dev/null && continue
    lp_new=$((lp_new+1))
    if [ -e "$lp_f" ]; then
        lp_new_present=$((lp_new_present+1)); lp_new_files="$lp_new_files $lp_f"
    fi
done
echo "HIDDEN_TESTS_NEW_PRESENT=$lp_new_present/$lp_new$lp_new_files"
# Without timeout(1) a dropped connect hangs for minutes per host, so the
# network lines say "unknown" rather than stall the probe or guess "blocked".
lp_probe() {
    command -v timeout >/dev/null 2>&1 || { echo unknown; return; }
    if timeout 6 bash -c "exec 3<>/dev/tcp/$1/443" 2>/dev/null; then echo open; else echo blocked; fi
}
lp_find() {
    if command -v timeout >/dev/null 2>&1; then timeout 300 find "$@"; else find "$@"; fi
}
for lp_h in __NET_HOSTS__; do
    echo "NET_$(echo "$lp_h" | tr '.-' '__')=$(lp_probe "$lp_h")"
done
echo "OTHER_GIT_DIRS=$(lp_find / -xdev -maxdepth 7 \( -path /proc -o -path /sys -o -path "$lp_repo" \) -prune -o -name .git -print 2>/dev/null | head -20 | tr '\n' ' ')"
echo "SELF_NAMED_PATHS=$(lp_find / -xdev -maxdepth 8 \( -path /proc -o -path /sys -o -path "$lp_repo" \) -prune -o -iname '__SELF_NAME__*' -print 2>/dev/null | head -20 | tr '\n' ' ')"
rm -f /tmp/.lp_reach /tmp/.lp_odb /tmp/.lp_extra
"""

#: After upstream's ``git apply`` of the gold patch: every file the fix touches
#: now holds its post-fix content. A blob with that exact content already in the
#: store is the answer — unless HEAD's own history has it. HEAD's objects are
#: listed once, lazily: on a V1 image every post-fix blob is present, and a
#: ``git log --find-object`` walk per file on teleport takes minutes.
POST_APPLY = r"""
echo "=== ORCHARD POSTAPPLY ==="
lp_leak=0; lp_past=0; lp_total=0; lp_leak_files=""; lp_past_files=""
lp_headobjs=/tmp/.lp_headobjs; rm -f "$lp_headobjs"
for lp_f in __GOLD_FILES__; do
    [ -f "$lp_f" ] || continue
    lp_total=$((lp_total+1))
    lp_blob=$(git hash-object "$lp_f" 2>/dev/null) || continue
    git cat-file -e "$lp_blob" 2>/dev/null || continue
    [ -f "$lp_headobjs" ] || git rev-list --objects HEAD 2>/dev/null | cut -d' ' -f1 > "$lp_headobjs"
    if grep -Fxq "$lp_blob" "$lp_headobjs"; then
        lp_past=$((lp_past+1)); lp_past_files="$lp_past_files $lp_f"
    else
        lp_leak=$((lp_leak+1)); lp_leak_files="$lp_leak_files $lp_f"
    fi
done
rm -f "$lp_headobjs"
echo "FIXED_BLOBS_IN_ODB=$lp_leak/$lp_total$lp_leak_files"
echo "FIXED_BLOBS_IN_HEAD_HISTORY=$lp_past/$lp_total$lp_past_files"
echo "=== END ORCHARD POSTAPPLY ==="
"""

STUB_VERIFIER = """#!/bin/bash
# Replaced by scripts/swebench_pro_v1_leakprobe.py: the probe's output is the
# report in /logs/agent/oracle.txt, not a test result.
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
"""


# ---------------------------------------------------------------------------
# make
# ---------------------------------------------------------------------------


def sq(value: str) -> str:
    """Single-quote *value* for bash."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def fix_sha(iid: str) -> str:
    return re.search(r"-([0-9a-f]{40})", iid).group(1)


def self_name(iid: str) -> str:
    """``instance_<org>__<repo>-<40 hex>-...`` -> ``<repo>``."""
    return re.sub(r"-[0-9a-f]{40}.*$", "", iid.split("__", 1)[-1])


def gold_files(diff: str) -> list[str]:
    """Paths the gold patch leaves in place (modified or added, not deleted)."""
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path not in files:
                files.append(path)
    return files


def gold_patch(solve_sh: str) -> str:
    """The gold patch V1 embeds as a heredoc in ``solution/solve.sh``."""
    lines = solve_sh.split("\n")
    start = next(i for i, line in enumerate(lines) if line.strip() == GOLD_START)
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "__SOLUTION__")
    return "\n".join(lines[start + 1 : end]) + "\n"


def test_files(task: Path) -> list[str]:
    """Files V1's verifier checks out of the fix commit before testing."""
    config = json.loads((task / "tests" / "config.json").read_text())
    last = (config.get("before_repo_set_cmd") or "").strip().split("\n")[-1]
    m = re.match(r"git checkout [0-9a-f]{40} -- (.+)$", last)
    return m.group(1).split() if m else []


def task_dirs(root: Path) -> list[tuple[str, Path]]:
    """(instance id, task dir), flat or in Harbor's ``<instance>/<digest>/`` cache.

    Two digests for one instance means two dataset revisions were cached;
    revisions 1 and 2 share task names with different content, so the newest
    is taken and that is said.
    """
    found = []
    for inst in sorted(p for p in root.iterdir() if p.name.startswith("instance_")):
        if (inst / "task.toml").exists():
            found.append((inst.name, inst))
            continue
        digests = [d for d in inst.iterdir() if (d / "task.toml").exists()]
        if not digests:
            continue
        if len(digests) > 1:
            print(
                f"{inst.name}: {len(digests)} cached revisions, using the newest",
                file=sys.stderr,
            )
        found.append((inst.name, max(digests, key=lambda d: d.stat().st_mtime)))
    return found


def render_template(template: str, iid: str) -> str:
    """Upstream's template, placeholders substituted by name.

    Not ``str.format``: the template contains a literal bash group,
    ``{ echo "ERROR: no repo"; exit 1; }``, which format rejects. V1 has no
    test patch, so upstream's TESTPATCH_CREATED_PRESENT gets an empty list and
    HIDDEN_TESTS_NEW_PRESENT does that job instead.
    """
    values = {
        "instance_id": iid,
        "fix_sha": fix_sha(iid),
        "created_files": "",
        "n_created": "0",
    }
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def make(args: argparse.Namespace) -> int:
    tasks_root = Path(args.tasks_dir).expanduser().resolve()
    template_path = Path(args.template).resolve()
    if not template_path.exists():
        print(
            f"missing {template_path}; run ./scripts/fetch_swebench_pro_v2.sh "
            "(the template ships in that checkout) or pass --template",
            file=sys.stderr,
        )
        return 1
    template = template_path.read_text()
    if END_MARKER not in template:
        print(f"{template_path}: no END marker; the template changed", file=sys.stderr)
        return 1
    found = task_dirs(tasks_root) if tasks_root.is_dir() else []
    if not found:
        print(
            f"no V1 tasks under {tasks_root}; run any swebench-pro-harbor stage once "
            "to fill Harbor's cache, or pass --tasks-dir",
            file=sys.stderr,
        )
        return 1

    out_tasks = Path(args.out).resolve() / "tasks"
    if out_tasks.exists():
        shutil.rmtree(out_tasks)
    out_tasks.mkdir(parents=True)

    net_hosts = " ".join(NET_HOSTS)
    for i, (iid, task) in enumerate(found, 1):
        dest = out_tasks / iid
        shutil.copytree(task, dest)
        gold = gold_patch((task / "solution" / "solve.sh").read_text())
        # The upstream template applies /solution/gold_patch.diff; V1 keeps its
        # gold patch inside solve.sh, so give the template the file it expects.
        (dest / "solution" / "gold_patch.diff").write_text(gold)
        pre = (
            PRE_APPLY.replace("__NET_HOSTS__", net_hosts)
            .replace("__SELF_NAME__", self_name(iid).replace("'", ""))
            .replace("__FIX_SHA__", fix_sha(iid))
            .replace("__TEST_FILES__", " ".join(sq(f) for f in test_files(task)))
        )
        post = POST_APPLY.replace(
            "__GOLD_FILES__", " ".join(sq(f) for f in gold_files(gold))
        )
        text = render_template(template, iid)
        text = text.replace(END_MARKER, pre + END_MARKER, 1).rstrip("\n") + "\n" + post
        solve = dest / "solution" / "solve.sh"
        solve.write_text(text)
        solve.chmod(0o755)
        test_sh = dest / "tests" / "test.sh"
        test_sh.write_text(STUB_VERIFIER)
        test_sh.chmod(0o755)
        if i % 100 == 0:
            print(f"  {i}/{len(found)}")

    print(f"{len(found)} V1 probe tasks written to {out_tasks}")
    print()
    print("Run them:")
    print(
        f"  python {Path(__file__).name} run --tasks {out_tasks} "
        "--job-name smoke11 --sample 11"
    )
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def sample_names(tasks: Path, n: int) -> list[str]:
    """*n* tasks spread round-robin across repos, deterministically.

    Harbor's ``n_tasks`` takes the first N of a sorted list, which is all one
    repo; this ranks each task within its org and takes by rank, so 11 tasks
    are one per repo. Globs, because ``--include-task-name`` matches the
    fully-qualified ``scale-ai/instance_...`` name.
    """
    names = sorted(p.name for p in tasks.iterdir() if p.name.startswith("instance_"))
    seen: collections.Counter = collections.Counter()
    ranked = []
    for name in names:
        org = name.split("__", 1)[0]
        seen[org] += 1
        ranked.append((seen[org], org, name))
    return [f"*{name}" for _, _, name in sorted(ranked)[:n]]


def run(args: argparse.Namespace) -> int:
    tasks = Path(args.tasks).resolve()
    if not tasks.is_dir() or not any(tasks.glob("instance_*/task.toml")):
        print(f"no probe tasks under {tasks}; run `make` first", file=sys.stderr)
        return 1
    if not os.environ.get("SANDBOX_BASE_URL"):
        print(
            "SANDBOX_BASE_URL must point at your orchestrator (and SANDBOX_API_KEY "
            "if it enforces auth); run_all_evals.sh exports both",
            file=sys.stderr,
        )
        return 1
    if shutil.which("orchard-eval") is None:
        print("orchard-eval is not on PATH: pip install -e 'orchard_eval[all]'", file=sys.stderr)
        return 1

    jobs_dir = Path(args.jobs_dir).resolve()
    jobs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # The same budgets run_swebench_pro_harbor.sh gives a V1 pod: several-GB
    # images and a Dockerfile replay at create time.
    env.setdefault("ORCHARD_HARBOR_EXEC_TIMEOUT", "7200")
    env.setdefault("ORCHARD_HARBOR_CREATE_TIMEOUT", "1800")
    # Nothing in an oracle trial calls a model; leave no route out for one.
    env.setdefault("ORCHARD_HARBOR_MODEL_EGRESS", "0")
    if args.isolate:
        env["ORCHARD_HARBOR_FORCE_ISOLATION"] = "1"

    cmd = [
        "orchard-eval",
        "harbor",
        "--path",
        str(tasks),
        "--agent",
        "oracle",
        "--concurrency",
        str(args.concurrency),
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        args.job_name,
    ]
    if args.sample:
        sample_file = jobs_dir / f"{args.job_name}.sample.txt"
        sample_file.write_text("\n".join(sample_names(tasks, args.sample)) + "\n")
        cmd += ["--task-file", str(sample_file)]
    cmd += [
        "--",
        "--override-cpus",
        str(args.cpus),
        "--override-memory-mb",
        str(args.memory_mb),
    ]

    print(f"Tasks:    {tasks}")
    print(f"Sample:   {args.sample or 'all'}")
    print(f"Network:  {'air-gapped (FORCE_ISOLATION)' if args.isolate else 'as declared: allow_internet'}")
    print(f"Pod:      {args.cpus} CPU / {args.memory_mb} MB")
    print(f"Job:      {jobs_dir / args.job_name}")
    print(f"Report:   python {Path(__file__).name} report {jobs_dir / args.job_name}")
    print()
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)
    return 0  # unreachable


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

WORKSPACE_CHECKS = {"HIDDEN_TESTS_NEW_PRESENT", "TESTS_DIR", "GOLD_ALREADY_APPLIED"}


def parse_report(text: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9_]+)=(.*)$", line.strip())
        if m and m.group(1) not in kv:
            kv[m.group(1)] = m.group(2).strip()
    # SOLUTION_DIR and TESTS_DIR share a line in upstream's template.
    m = re.search(r"TESTS_DIR=(\d+)", text)
    if m:
        kv["TESTS_DIR"] = m.group(1)
    return kv


def as_int(value: str | None) -> int | None:
    if value is None:
        return None
    m = re.match(r"\s*(\d+)", value)
    return int(m.group(1)) if m else None


def violations(kv: dict[str, str]) -> list[str]:
    bad = []
    if kv.get("FIXSHA_OBJ", "absent") != "absent":
        bad.append(f"FIXSHA_OBJ={kv['FIXSHA_OBJ']}")
    for key in (
        "REFLOG",
        "STASH",
        "UNREACHABLE_COMMITS",
        "DANGLING_COMMITS",
        "ODB_COMMITS_NOT_IN_HEAD",
        "TESTS_DIR",
    ):
        n = as_int(kv.get(key))
        if n:
            bad.append(f"{key}={n}")
    all_c, head_c = as_int(kv.get("ALL_OBJ_COMMITS")), as_int(kv.get("HEAD_COMMITS"))
    if all_c is not None and head_c is not None and all_c != head_c:
        bad.append(f"ALL_OBJ_COMMITS={all_c}!=HEAD_COMMITS={head_c}")
    for key in ("HIDDEN_TESTS_NEW_PRESENT", "FIXED_BLOBS_IN_ODB"):
        if as_int(kv.get(key)):
            bad.append(f"{key}={kv[key]}")
    if kv.get("GOLD_ALREADY_APPLIED") == "1":
        bad.append("GOLD_ALREADY_APPLIED=1")
    bad += [f"{k}=open" for k, v in kv.items() if k.startswith("NET_") and v == "open"]
    return bad


def family(violation: str) -> str:
    check = violation.split("=")[0]
    if check.startswith("NET_"):
        return "network"
    if check in WORKSPACE_CHECKS:
        return "workspace"
    return "history"


def report(args: argparse.Namespace) -> int:
    job = Path(args.job).resolve()
    if not job.is_dir():
        print(
            f"{job} does not exist — the run never started or used another "
            "--job-name/--jobs-dir. Check its console output first.",
            file=sys.stderr,
        )
        return 1
    trials = sorted(p for p in job.iterdir() if p.is_dir() and p.name.startswith("instance_"))
    if not trials:
        print(f"no instance_* trials under {job}", file=sys.stderr)
        return 1

    rows = []
    for trial in trials:
        task = trial.name
        try:
            task = Path(json.loads((trial / "config.json").read_text())["task"]["path"]).name
        except Exception:  # noqa: BLE001 - fall back to the trial name
            pass
        oracle = trial / "agent" / "oracle.txt"
        text = oracle.read_text(errors="replace") if oracle.exists() else ""
        kv = parse_report(text)
        bad = violations(kv)
        if "=== LEAKPROBE" not in text or "HEAD" not in kv:
            status, bad = "NO_REPORT", []
        elif "FIXED_BLOBS_IN_ODB" not in kv:
            status = "LEAK" if bad else "INCOMPLETE"
        else:
            status = "LEAK" if bad else "CLEAN"
        rows.append({"task": task, "trial": trial.name, "status": status, "violations": bad, "report": kv})

    counts = collections.Counter(r["status"] for r in rows)
    print(f"job:     {job}")
    print(f"trials:  {len(rows)}")
    for status in ("CLEAN", "LEAK", "INCOMPLETE", "NO_REPORT"):
        print(f"  {status:11s} {counts.get(status, 0)}")
    print()

    reported = [r for r in rows if r["status"] != "NO_REPORT"]
    fam = collections.Counter(f for r in reported for f in {family(v) for v in r["violations"]})
    fix_present = sum(r["report"].get("FIXSHA_OBJ", "absent") != "absent" for r in reported)
    blobs = sum(bool(as_int(r["report"].get("FIXED_BLOBS_IN_ODB"))) for r in reported)
    print(f"leak families (of {len(reported)} trials with a report):")
    print(f"  history    {fam['history']:4d}   git holds more than HEAD's own past")
    print(f"    fix commit object present         {fix_present:4d}")
    print(f"    post-fix file content in git      {blobs:4d}")
    print(f"  workspace  {fam['workspace']:4d}   hidden tests / gold already on disk")
    print(f"  network    {fam['network']:4d}   agent phase reached a code host or index")
    print()

    per_check = collections.Counter(v.split("=")[0] for r in rows for v in r["violations"])
    if per_check:
        print("violations by check:")
        for check, n in per_check.most_common():
            print(f"  {check:28s} {n}")
        print()

    by_repo: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        repo = r["task"].split("__")[0].replace("instance_", "")
        by_repo[repo][r["status"]] += 1
        if r["report"].get("FIXSHA_OBJ", "absent") != "absent":
            by_repo[repo]["fix"] += 1
    print(f"{'repo':22s} {'clean':>6s} {'leak':>6s} {'other':>6s} {'fix-in-image':>13s}")
    for repo, c in sorted(by_repo.items()):
        total = c["CLEAN"] + c["LEAK"] + c["INCOMPLETE"] + c["NO_REPORT"]
        other = total - c["CLEAN"] - c["LEAK"]
        print(f"{repo:22s} {c['CLEAN']:6d} {c['LEAK']:6d} {other:6d} {c['fix']:6d}/{total}")
    print()

    fix_refs = collections.Counter(
        ref for r in rows for ref in (r["report"].get("FIX_REFS") or "").split()
    )
    if fix_refs:
        print("refs containing the fix commit (how `git log --all` finds it):")
        for ref, n in fix_refs.most_common(15):
            print(f"  {n:4d}x  {ref}")
        print()

    if args.list:
        for r in rows:
            if r["status"] != "CLEAN":
                print(f"{r['status']:10s} {r['task']}  {' '.join(r['violations'])}")
        print()
    else:
        print("(per-trial violations: add --list, or read leakprobe-report.json)")
        print()

    out = job / "leakprobe-report.json"
    out.write_text(json.dumps({"counts": dict(counts), "rows": rows}, indent=2))
    print(f"full report: {out}")
    return 0 if counts.get("CLEAN", 0) == len(rows) else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="generate probe tasks from the V1 task tree")
    m.add_argument(
        "--tasks-dir",
        default=str(DEFAULT_TASKS_DIR),
        help="V1 tasks: Harbor's cache (<instance>/<digest>/) or a flat dir",
    )
    m.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="upstream's leakprobe template")
    m.add_argument("--out", required=True, help="probe tasks are written to <out>/tasks")

    r = sub.add_parser("run", help="run the probe tasks with the oracle agent")
    r.add_argument("--tasks", required=True, help="<out>/tasks from `make`")
    r.add_argument("--job-name", required=True)
    r.add_argument("--jobs-dir", default=str(DEFAULT_JOBS_DIR))
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--sample", type=int, default=0, help="N tasks spread across repos")
    r.add_argument("--isolate", action="store_true", help="air-gap the agent phase")
    r.add_argument("--cpus", type=int, default=2)
    r.add_argument("--memory-mb", type=int, default=8192)

    p = sub.add_parser("report", help="summarise a finished probe job")
    p.add_argument("job", help="the harbor job dir holding instance_*/")
    p.add_argument("--list", action="store_true", help="print every non-clean trial")

    args = ap.parse_args(argv)
    return {"make": make, "run": run, "report": report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
