#!/usr/bin/env bash
# Fetch and verify the SWE-bench Pro V2 task tree.
#
# V2 is not on the Harbor registry. Upstream ships it as 642 Harbor task
# directories inside its own git repository, so unlike every other benchmark
# here there is no `--dataset` to name: the tasks arrive as files and
# `orchard-eval harbor --path` points at them.
#
# That makes this script part of the grader, not a convenience. The task
# directories *are* the verifier — `tests/test.sh`, `tests/config.json` and
# `tests/test_patch.patch` decide what passes — so the tree has to be pinned
# and checked rather than tracked. Two things follow:
#
#   1. A commit is pinned, not a branch. Same reasoning as
#      `run_swebench_pro_harbor.sh` pinning `@2` rather than `@latest`: a score
#      is only comparable to another score measured against the same grader,
#      and `main` moving under a long sweep is not something the results would
#      record.
#   2. The checksums are verified. Upstream ships `v2/SHA256SUMS` covering
#      every file for exactly this purpose, and a task tree that has drifted —
#      a half-finished clone, a partially-applied patch — fails as a model
#      scoring badly rather than as an error.
#
# Why V2 at all, when `swebench-pro-harbor` already measures 731 tasks: V1's
# images carry the fixing commit in `/app`'s git history and the task name
# contains its SHA, so an agent can read the answer. Audited on the 2026-09-21
# sweep, 701 of 731 trials did read that history and 219 copied gold straight
# into a non-test source file, scoring 90% against 71% for the trials that
# never saw it — 5 to 6 points of inflation, as a floor. V2 rebuilds every
# image from a sanitised bundle with no fixing commit, stray refs, stashes or
# hooks. See "SWE-bench Pro V2, via Harbor" in the README.
#
# Usage:
#   ./scripts/fetch_swebench_pro_v2.sh              # clone/update and verify
#   PRO_V2_DIR=/data/sbp-v2 ./scripts/fetch_swebench_pro_v2.sh
#   PRO_V2_REF=main ./scripts/fetch_swebench_pro_v2.sh   # deliberately unpinned
#
# Environment:
#   PRO_V2_DIR   Where the checkout lives. Default ./third_party/SWE-bench_Pro-os
#   PRO_V2_REF   Commit/tag to check out. Defaults to the pin below.
#   SKIP_CHECKSUMS=1
#                Skip `shasum -c`. For a slow filesystem when you have already
#                verified the same tree once, and for nothing else.
set -euo pipefail

REPO_URL="https://github.com/scaleapi/SWE-bench_Pro-os"

# v2.0.0 — "SWE-bench Pro V2 (v2.0.0): 642 Harbor tasks, HARD-51 subset",
# merged 2026-09-22. Bump this deliberately, and re-run the ceiling and floor
# checks afterwards: a new pin is a new grader.
PRO_V2_REF="${PRO_V2_REF:-66f92766bba642462d4bbe5479e83f91f9211862}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRO_V2_DIR="${PRO_V2_DIR:-${SCRIPT_DIR}/../third_party/SWE-bench_Pro-os}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is not on PATH." >&2
    exit 1
fi

# `shasum` on macOS, `sha256sum` on most Linux images. Both read the same
# `SHA256SUMS` format, so either will do.
CHECKSUM_CMD=""
if command -v shasum >/dev/null 2>&1; then
    CHECKSUM_CMD="shasum -a 256 -c"
elif command -v sha256sum >/dev/null 2>&1; then
    CHECKSUM_CMD="sha256sum -c"
fi

if [[ ! -d "${PRO_V2_DIR}/.git" ]]; then
    echo "Cloning ${REPO_URL}"
    echo "     -> ${PRO_V2_DIR}"
    mkdir -p "$(dirname "${PRO_V2_DIR}")"
    # Not --depth 1: a pinned commit is not necessarily the tip, and a shallow
    # clone cannot check one out without a second fetch anyway.
    git clone --quiet "${REPO_URL}" "${PRO_V2_DIR}"
else
    echo "Updating ${PRO_V2_DIR}"
    git -C "${PRO_V2_DIR}" fetch --quiet origin
fi

git -C "${PRO_V2_DIR}" checkout --quiet "${PRO_V2_REF}" || {
    echo "Could not check out '${PRO_V2_REF}'. If you bumped PRO_V2_REF, make" >&2
    echo "sure it exists upstream: git -C '${PRO_V2_DIR}' fetch origin" >&2
    exit 1
}

PRO_V2_DIR="$(cd "${PRO_V2_DIR}" && pwd)"
TASKS_DIR="${PRO_V2_DIR}/v2/tasks"

if [[ ! -d "${TASKS_DIR}" ]]; then
    echo "No v2/tasks in ${PRO_V2_DIR} at ${PRO_V2_REF}." >&2
    echo "That ref predates the V2 release; use the default pin." >&2
    exit 1
fi

if [[ "${SKIP_CHECKSUMS:-0}" == "1" ]]; then
    echo "Checksums:   skipped (SKIP_CHECKSUMS=1)"
elif [[ -z "${CHECKSUM_CMD}" ]]; then
    echo "Checksums:   SKIPPED — neither shasum nor sha256sum is on PATH." >&2
    echo "             The task tree is the grader; verify it before trusting a score." >&2
elif [[ ! -f "${PRO_V2_DIR}/v2/SHA256SUMS" ]]; then
    echo "Checksums:   SKIPPED — no v2/SHA256SUMS at this ref." >&2
else
    echo "Verifying v2/SHA256SUMS (this walks every task file; ~a minute)"
    if ! (cd "${PRO_V2_DIR}/v2" && ${CHECKSUM_CMD} SHA256SUMS >/dev/null); then
        echo >&2
        echo "CHECKSUM MISMATCH in ${PRO_V2_DIR}/v2." >&2
        echo "The task tree is the verifier, so do not run against it. Re-clone:" >&2
        echo "  rm -rf '${PRO_V2_DIR}' && $0" >&2
        exit 1
    fi
    echo "Checksums:   OK"
fi

TASK_COUNT="$(find "${TASKS_DIR}" -mindepth 2 -maxdepth 2 -name task.toml | wc -l | tr -d ' ')"

echo
echo "Checkout:    ${PRO_V2_DIR}"
echo "Ref:         ${PRO_V2_REF}"
echo "Tasks:       ${TASK_COUNT}  (upstream ships 642)"
echo "HARD-51:     ${PRO_V2_DIR}/v2/hard51_ids.txt"
echo "Tooling:     ${PRO_V2_DIR}/v2/tooling  (patch_replay, probes)"
echo
if [[ "${TASK_COUNT}" != "642" ]]; then
    echo "NOTE: expected 642 task directories. Anything else means the ref is not" >&2
    echo "      the V2 release, or the checkout is incomplete." >&2
fi
echo "Export this for the run scripts and run_all_evals.sh:"
echo "  export PRO_V2_DIR=${PRO_V2_DIR}"
echo
echo "Then confirm the provider understands every task, offline and with no pods:"
echo "  harbor-orchard audit ${TASKS_DIR}"
