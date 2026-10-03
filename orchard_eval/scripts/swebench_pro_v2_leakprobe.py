#!/usr/bin/env python3
"""Probe every SWE-bench Pro V2 image for git-history and hidden-file leaks.

V2's claim is that its images carry a sanitised history — no fixing commit, no
stray refs, stashes or hooks — and that the agent phase is air-gapped. This
checks both claims on *this* cluster rather than taking them on trust, using
upstream's own probe template (``v2/tooling/probe/leakprobe_solve.sh.tmpl``)
and a few checks it does not make. Upstream's generator for it,
``make_probe_task.py --kind leak``, cannot render a single task at the pinned
commit (see ``render_upstream_leakprobe``), so this renders the template itself.

For V1 (``scale-ai/swe-bench-pro@2``) use scripts/swebench_pro_v1_leakprobe.py,
which renders the same template with the same checks.

The probe is a Harbor task whose ``solve.sh`` reports on the sandbox instead of
solving it, so running it with ``-a oracle`` puts the report in each trial's
``agent/oracle.txt`` — written during the agent phase, with the agent's network
policy and the agent's view of the filesystem.

Three steps:

    python scripts/swebench_pro_v2_leakprobe.py make --out /tmp/sbp-v2-leakprobe
    PRO_V2_DIR=/tmp/sbp-v2-leakprobe JOB_NAME=leakprobe-$(date +%Y%m%d-%H%M%S) \\
        ./scripts/run_swebench_pro_v2.sh oracle 64
    python scripts/swebench_pro_v2_leakprobe.py report results/harbor/leakprobe-...

``SAMPLE=N`` / ``HARD51=1`` work on the middle step as they do for a real run.

Checks, all run inside the agent-phase sandbox. A trial FAILS on any of:

  upstream's
    FIXSHA_OBJ != absent        the fixing commit (SHA in the task name) exists
    REFLOG, STASH > 0           reflog / stash entries survived
    UNREACHABLE_COMMITS > 0     commits reachable from nothing (``git fsck``)
    DANGLING_COMMITS > 0        same, ignoring reflogs
    ALL_OBJ_COMMITS != HEAD_COMMITS
                                some ref points at history HEAD does not have
    TESTPATCH_CREATED_PRESENT   files the hidden test patch creates already exist
    TESTS_DIR != 0              /tests is visible to the agent
  added here
    ODB_COMMITS_NOT_IN_HEAD > 0 commit objects anywhere in the object store —
                                packed, loose or via alternates — that HEAD
                                cannot reach. Catches what refs-based checks miss.
    FIXED_BLOBS_IN_ODB > 0      a file's *post-fix* content is already stored as
                                a blob that HEAD's history does not contain,
                                i.e. the answer is recoverable without any
                                commit at all. Post-fix content that HEAD's own
                                history has (the fix restores an old version) is
                                reported as FIXED_BLOBS_IN_HEAD_HISTORY instead:
                                the past is fair game, so that is not a failure.
    GOLD_ALREADY_APPLIED = 1    the working tree already contains the fix
    NET_<host> = open           the agent phase could reach a code host or
                                package index

The report groups these into three families, because they answer different
questions: *history* (FIXSHA_OBJ, REFLOG, STASH, UNREACHABLE/DANGLING_COMMITS,
ALL_OBJ_COMMITS, ODB_COMMITS_NOT_IN_HEAD, FIXED_BLOBS_IN_ODB — can the answer be
read out of git?), *workspace* (hidden tests, /tests, gold already applied) and
*network*.

Informational only (printed, never a failure): REFS, PACKED_REFS, HOOKS,
ORIG_FETCH_HEAD, FIX_REFS, FIXED_BLOBS_IN_HEAD_HISTORY, OTHER_GIT_DIRS,
SELF_NAMED_PATHS. FIX_REFS names the refs that contain the fix commit when it
is present (e.g. ``refs/remotes/origin/master``), i.e. how an agent would find
it with ``git log --all``. SELF_NAMED_PATHS lists paths outside the repo named
after the project — an installed copy of a *newer* release in site-packages or
a Go module cache is the residual leak the history checks cannot see, and these
need a human to read.

By default the verifier is replaced with one that writes reward 1 immediately:
the report is the output and the real test suite would only add an hour of pod
time. ``--keep-verifier`` keeps it, which additionally re-proves the oracle.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRO_V2_DIR = SCRIPT_DIR.parent / "third_party" / "SWE-bench_Pro-os"

END_MARKER = 'echo "=== END LEAKPROBE ==="'

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

#: Runs before upstream's END marker, i.e. before the gold patch is applied.
#: Plain bash plus coreutils/busybox and git, because some images have nothing
#: else — no curl, no python.
PRE_APPLY = r"""
# --- orchard_eval additions (scripts/swebench_pro_v2_leakprobe.py) ---
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
    echo "ODB_EXTRA_SAMPLE=$(head -5 /tmp/.lp_extra | git log --no-walk=unsorted --stdin --format='%h %cI %s' 2>/dev/null | tr '\n' '|' | cut -c1-400)"
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

#: Appended after upstream's `git apply` of the gold patch. Every file the fix
#: touches now holds its post-fix content; if that exact blob is already in the
#: object store, the answer was sitting in the image — unless HEAD's own
#: history contains it, i.e. the fix restores a version the file had before
#: (a revert, a reapplied change). That past is legitimately visible, so those
#: are reported separately, and only blobs outside HEAD's history count as a
#: leak. HEAD's objects are listed once, lazily, rather than with a
#: ``git log --find-object`` per file: on an image that does carry the fix,
#: every post-fix blob is present, and a full-history diff walk per file on
#: teleport takes minutes.
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
# Replaced by scripts/swebench_pro_v2_leakprobe.py: the probe's output is the
# report in /logs/agent/oracle.txt, not a test result.
mkdir -p /logs/verifier
echo 1 > /logs/verifier/reward.txt
"""


def sq(value: str) -> str:
    """Single-quote *value* for bash."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def self_name(instance_id: str) -> str:
    """``instance_<org>__<repo>-<40 hex>-...`` -> ``<repo>``."""
    repo = instance_id.split("__", 1)[-1]
    return re.sub(r"-[0-9a-f]{40}.*$", "", repo)


def gold_files(diff: str) -> list[str]:
    """Paths the gold patch leaves in place (modified or added, not deleted)."""
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :].strip()
            if path not in files:
                files.append(path)
    return files


def render_upstream_leakprobe(template: str, iid: str, created: list[str]) -> str:
    """Upstream's ``make_probe_task.py --kind leak``, minus its two bugs.

    At the pinned commit that script calls ``.format()`` on the template's
    ``Path`` rather than on its text, so ``--kind leak`` raises AttributeError
    on every task. Fixing that alone is not enough: the template contains a
    literal bash group, ``{ echo "ERROR: no repo"; exit 1; }``, which
    ``str.format`` reads as a field and rejects with KeyError. So the four
    placeholders are substituted by name instead. Their values and the
    template itself are upstream's.
    """
    values = {
        "instance_id": iid,
        "fix_sha": fix_sha(iid),
        "created_files": " ".join(f"'{c}'" for c in created),
        "n_created": str(len(created)),
    }
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def fix_sha(iid: str) -> str:
    return re.search(r"-([0-9a-f]{40})", iid).group(1)


def v2_created_files(task: Path) -> list[str]:
    """Files V2's hidden test patch creates, exactly as upstream derives them."""
    tp = (task / "tests" / "test_patch.patch").read_text().split("\n")
    created = [
        tp[i + 1][6:]
        for i, line in enumerate(tp[:-1])
        if line.startswith("--- /dev/null") and tp[i + 1].startswith("+++ b/")
    ]
    created += [line[len("rename to ") :] for line in tp if line.startswith("rename to ")]
    return created


def make(args: argparse.Namespace) -> int:
    pro_v2 = Path(args.pro_v2_dir).resolve()
    tasks = pro_v2 / "v2" / "tasks"
    template_path = pro_v2 / "v2" / "tooling" / "probe" / "leakprobe_solve.sh.tmpl"
    for p in (tasks, template_path):
        if not p.exists():
            print(
                f"missing {p}; run ./scripts/fetch_swebench_pro_v2.sh first",
                file=sys.stderr,
            )
            return 1
    template = template_path.read_text()
    if END_MARKER not in template:
        print(f"{template_path}: no END marker; the template changed", file=sys.stderr)
        return 1

    out_root = Path(args.out).resolve()
    out_tasks = out_root / "v2" / "tasks"
    if out_tasks.exists():
        shutil.rmtree(out_tasks)
    out_tasks.mkdir(parents=True)
    hard51 = pro_v2 / "v2" / "hard51_ids.txt"
    if hard51.exists():
        shutil.copy2(hard51, out_root / "v2" / "hard51_ids.txt")

    net_hosts = " ".join(NET_HOSTS)
    task_dirs = sorted(
        p for p in tasks.iterdir() if p.is_dir() and p.name.startswith("instance_")
    )
    for i, task in enumerate(task_dirs, 1):
        iid = task.name
        dest = out_tasks / iid
        shutil.copytree(task, dest)
        pre = (
            PRE_APPLY.replace("__NET_HOSTS__", net_hosts)
            .replace("__SELF_NAME__", self_name(iid).replace("'", ""))
            .replace("__FIX_SHA__", fix_sha(iid))
        )
        gold = (task / "solution" / "gold_patch.diff").read_text()
        text = render_upstream_leakprobe(template, iid, v2_created_files(task))
        post = POST_APPLY.replace(
            "__GOLD_FILES__", " ".join(sq(f) for f in gold_files(gold))
        )
        text = text.replace(END_MARKER, pre + END_MARKER, 1).rstrip("\n") + "\n" + post
        solve = dest / "solution" / "solve.sh"
        solve.write_text(text)
        solve.chmod(0o755)
        if not args.keep_verifier:
            test_sh = dest / "tests" / "test.sh"
            test_sh.write_text(STUB_VERIFIER)
            test_sh.chmod(0o755)
        if i % 100 == 0:
            print(f"  {i}/{len(task_dirs)}")

    print(f"{len(task_dirs)} probe tasks written to {out_tasks}")
    print()
    print("Run them (the agent must be oracle — the probe *is* the solution):")
    print(f"  PRO_V2_DIR={out_root} JOB_NAME=leakprobe-$(date +%Y%m%d-%H%M%S) \\")
    print(
        "      OVERRIDE_CPUS=2 OVERRIDE_MEMORY_MB=8192 "
        "./scripts/run_swebench_pro_v2.sh oracle 64"
    )
    return 0


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
    for key in ("TESTPATCH_CREATED_PRESENT", "FIXED_BLOBS_IN_ODB"):
        if as_int(kv.get(key)):
            bad.append(f"{key}={kv[key]}")
    if kv.get("GOLD_ALREADY_APPLIED") == "1":
        bad.append("GOLD_ALREADY_APPLIED=1")
    bad += [f"{k}=open" for k, v in kv.items() if k.startswith("NET_") and v == "open"]
    return bad


WORKSPACE_CHECKS = {"TESTPATCH_CREATED_PRESENT", "TESTS_DIR", "GOLD_ALREADY_APPLIED"}


def family(violation: str) -> str:
    """history / workspace / network — the question a violation answers."""
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
            f"{job} does not exist — the oracle run never started or used another "
            "JOB_NAME/JOBS_DIR. Check its console output first.",
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
        if "=== LEAKPROBE" not in text or "HEAD" not in kv:
            status, bad = "NO_REPORT", []
        elif "FIXED_BLOBS_IN_ODB" not in kv or "ODB_COMMITS_NOT_IN_HEAD" not in kv:
            status, bad = "INCOMPLETE", violations(kv)
        else:
            bad = violations(kv)
            status = "LEAK" if bad else "CLEAN"
        if status == "INCOMPLETE" and bad:
            status = "LEAK"
        rows.append({"task": task, "trial": trial.name, "status": status, "violations": bad, "report": kv})

    counts = collections.Counter(r["status"] for r in rows)
    print(f"job:     {job}")
    print(f"trials:  {len(rows)}")
    for status in ("CLEAN", "LEAK", "INCOMPLETE", "NO_REPORT"):
        print(f"  {status:11s} {counts.get(status, 0)}")
    print()

    per_check = collections.Counter(v.split("=")[0] for r in rows for v in r["violations"])
    if per_check:
        print("violations by check:")
        for check, n in per_check.most_common():
            print(f"  {check:28s} {n}")
        print()

    # The headline for "is the answer in the image": the history family, and
    # within it the two checks that are the answer rather than a symptom.
    reported = [r for r in rows if r["status"] != "NO_REPORT"]
    n = len(reported)
    fam = collections.Counter(
        f for r in reported for f in {family(v) for v in r["violations"]}
    )
    fix_present = sum(r["report"].get("FIXSHA_OBJ", "absent") != "absent" for r in reported)
    blobs = sum(bool(as_int(r["report"].get("FIXED_BLOBS_IN_ODB"))) for r in reported)
    print(f"leak families (of {n} trials with a report):")
    print(f"  history    {fam['history']:4d}   git holds more than HEAD's own past")
    print(f"    fix commit object present         {fix_present:4d}")
    print(f"    post-fix file content in git      {blobs:4d}")
    print(f"  workspace  {fam['workspace']:4d}   hidden tests / gold already on disk")
    print(f"  network    {fam['network']:4d}   agent phase reached a code host or index")
    print()

    by_repo: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        by_repo[r["task"].split("__")[0].replace("instance_", "")][r["status"]] += 1
    print(f"{'repo':22s} {'clean':>6s} {'leak':>6s} {'other':>6s}")
    for repo, c in sorted(by_repo.items()):
        other = sum(c.values()) - c["CLEAN"] - c["LEAK"]
        print(f"{repo:22s} {c['CLEAN']:6d} {c['LEAK']:6d} {other:6d}")
    print()

    for r in rows:
        if r["status"] != "CLEAN":
            print(f"{r['status']:10s} {r['task']}  {' '.join(r['violations'])}")
    if counts.get("CLEAN", 0) != len(rows):
        print()

    # Informational: paths a human should read, collapsed across trials.
    info = collections.Counter()
    for r in rows:
        for key in ("OTHER_GIT_DIRS", "SELF_NAMED_PATHS"):
            for path in (r["report"].get(key) or "").split():
                info[(key, re.sub(r"[0-9a-f]{7,}", "<sha>", path))] += 1
    if info:
        print("informational paths (not failures — check none is a newer copy of the project):")
        for (key, path), n in sorted(info.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {n:4d}x  {key:17s} {path}")
        print()

    past = [r for r in rows if as_int(r["report"].get("FIXED_BLOBS_IN_HEAD_HISTORY"))]
    if past:
        print(
            "post-fix content already in HEAD's own history (the fix restores an "
            "earlier version; visible to the agent via git log, not a leak):"
        )
        for r in past:
            print(f"  {r['task']}  {r['report']['FIXED_BLOBS_IN_HEAD_HISTORY']}")
        print()

    # Where the fix commit is reachable from, when it is present: the route an
    # agent takes to it (`git log --all`, `git show <sha>`).
    fix_refs = collections.Counter(
        ref for r in rows for ref in (r["report"].get("FIX_REFS") or "").split()
    )
    if fix_refs:
        print("refs containing the fix commit (how `git log --all` finds it):")
        for ref, n in fix_refs.most_common(15):
            print(f"  {n:4d}x  {ref}")
        print()

    out = job / "leakprobe-report.json"
    out.write_text(json.dumps({"counts": dict(counts), "rows": rows}, indent=2))
    print(f"full report: {out}")
    return 0 if counts.get("CLEAN", 0) == len(rows) else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="generate probe tasks from the V2 task tree")
    m.add_argument("--pro-v2-dir", default=str(DEFAULT_PRO_V2_DIR))
    m.add_argument("--out", required=True, help="written as <out>/v2/tasks, the layout PRO_V2_DIR expects")
    m.add_argument("--keep-verifier", action="store_true", help="run the real test suite too")
    r = sub.add_parser("report", help="summarise a finished oracle job over the probe tasks")
    r.add_argument("job", help="the harbor job dir holding instance_*/")
    args = ap.parse_args(argv)
    return make(args) if args.cmd == "make" else report(args)


if __name__ == "__main__":
    sys.exit(main())
