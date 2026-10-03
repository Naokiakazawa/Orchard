"""Shell fragments the sandbox runs on our behalf.

Two things Docker does for free have to be written out by hand once there is no
image builder: the exact file-placement rules of ``COPY``, and running a step as
a non-root ``USER``. Both live here so there is a single place to reason about
them, and so their text can be asserted on in tests without a cluster.
"""

from __future__ import annotations

import shlex

#: Name a whole build context is staged under. ``COPY . /app`` must place the
#: context's *contents* in ``/app``, which is the same rule as any other
#: directory source — so the context is staged as an ordinary directory and the
#: normal rule applies, rather than being special-cased.
CONTEXT_ROOT_NAME = "__hb_context__"


#: Docker's placement rules, in POSIX shell.
#:
#: The rules being reproduced, in the order they are decided:
#:   * a destination ending in ``/``, or more than one source, means the
#:     destination is a directory and is created if missing;
#:   * an existing directory destination is a directory regardless of spelling;
#:   * a **directory** source contributes its *contents*, never itself — this is
#:     the rule that surprises people, and the one ``COPY data /app/data/``
#:     depends on;
#:   * otherwise a single file source *becomes* the destination path.
COPY_ROUTINE = r"""
set -e
__hb_stage="$1"
__hb_dest="$2"
__hb_targets=$(mktemp)
trap 'rm -f "$__hb_targets"' EXIT

__hb_count=$(find "$__hb_stage" -mindepth 1 -maxdepth 1 | wc -l)
if [ "$__hb_count" -eq 0 ]; then
  echo "harbor-orchard: nothing staged for COPY" >&2
  exit 1
fi

case "$__hb_dest" in
  */) __hb_want_dir=1 ;;
  *)  __hb_want_dir=0 ;;
esac
[ "$__hb_count" -gt 1 ] && __hb_want_dir=1

__hb_clean=$(printf '%s' "$__hb_dest" | sed 's:/\{1,\}$::')
[ -z "$__hb_clean" ] && __hb_clean=/
[ -d "$__hb_clean" ] && __hb_want_dir=1

[ "$__hb_want_dir" = 1 ] && mkdir -p "$__hb_clean"

__hb_list=$(find "$__hb_stage" -mindepth 1 -maxdepth 1)
__hb_old_ifs=$IFS
IFS='
'
for __hb_src in $__hb_list; do
  IFS=$__hb_old_ifs
  if [ -d "$__hb_src" ] && [ ! -L "$__hb_src" ]; then
    mkdir -p "$__hb_clean"
    cp -a "$__hb_src"/. "$__hb_clean"/
    ( cd "$__hb_src" && find . -mindepth 1 -maxdepth 1 ) | while IFS= read -r __hb_entry; do
      printf '%s\n' "$__hb_clean/${__hb_entry#./}" >> "$__hb_targets"
    done
  elif [ "$__hb_want_dir" = 1 ]; then
    cp -a "$__hb_src" "$__hb_clean"/
    printf '%s\n' "$__hb_clean/$(basename "$__hb_src")" >> "$__hb_targets"
  else
    mkdir -p "$(dirname "$__hb_clean")"
    cp -a "$__hb_src" "$__hb_clean"
    printf '%s\n' "$__hb_clean" >> "$__hb_targets"
  fi
  IFS='
'
done
IFS=$__hb_old_ifs

if [ -n "${__HB_CHOWN:-}" ] || [ -n "${__HB_CHMOD:-}" ]; then
  while IFS= read -r __hb_target; do
    [ -n "$__hb_target" ] || continue
    [ -n "${__HB_CHOWN:-}" ] && chown -R "$__HB_CHOWN" "$__hb_target"
    [ -n "${__HB_CHMOD:-}" ] && chmod -R "$__HB_CHMOD" "$__hb_target"
  done < "$__hb_targets"
fi
"""


#: Unpack ``ADD``'s auto-extracted archives in place, as Docker does for local
#: tarballs (and, deliberately, not for downloaded ones).
EXTRACT_ROUTINE = r"""
set -e
__hb_stage="$1"
for __hb_entry in "$__hb_stage"/*; do
  [ -f "$__hb_entry" ] || continue
  case "$__hb_entry" in
    *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tbz2|*.tar.xz|*.txz|*.tar.zst)
      __hb_out="$__hb_entry.__hb_unpacked"
      mkdir -p "$__hb_out"
      tar -xf "$__hb_entry" -C "$__hb_out"
      rm -f "$__hb_entry"
      mv "$__hb_out" "$__hb_entry"
      ;;
  esac
done
"""


def script_invocation(routine: str, *args: str) -> str:
    """Render *routine* as a runnable ``sh`` command with positional arguments.

    The routine is passed to ``sh -c`` as one argument and the parameters after
    it, so nothing in a path can be re-interpreted as shell syntax — a task
    directory containing a space or a ``$`` stays a plain path.
    """
    quoted = " ".join(shlex.quote(argument) for argument in args)
    return f"sh -c {shlex.quote(routine)} __hb {quoted}".rstrip()


def wrap_user(command: str, user: str | None) -> str:
    """Run *command* as *user*, or unchanged when it is already root.

    ``su -m`` is the portable choice: it keeps the environment we passed in,
    which is where the replayed ``ENV`` values live. Dropping them would leave a
    ``USER agent`` step running with the base image's PATH instead of the
    Dockerfile's.
    """
    if not user or user in ("root", "0", "root:root"):
        return command
    name = user.split(":", 1)[0]
    return f"su -m -s /bin/sh -c {shlex.quote(command)} {shlex.quote(name)}"


def export_env(env: dict[str, str]) -> str:
    """Render *env* as ``export`` lines for a persisted profile script."""
    return "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())
    )


#: Launch *binary* bounded by ``$ORCHARD_AGENT_DEADLINE``, when the environment
#: set one and this image has a ``timeout``.
#:
#: Harbor's agent deadline is an ``asyncio.wait_for`` around the agent
#: coroutine: it cancels the await and nothing else, so the CLI keeps running
#: in the pod, where it blocks log collection and the verifier behind it and
#: goes on spending GPU until the pod is reaped. Bounding the CLI itself is
#: what makes that deadline real — ``settings.agent_deadline_margin`` carries
#: the measurements.
#:
#: The wrapper goes around the CLI *only*, never the pipeline Harbor builds
#: around it (``... | tee /logs/agent/...``): when the deadline fires the CLI
#: dies, its pipe closes, and ``tee`` drains and flushes normally instead of
#: being killed mid-write. Keeping the transcript readable is the whole reason
#: for stopping early rather than letting the exec deadline take the pod down
#: with the transcript still in it.
#:
#: ``-k N`` and a positional duration rather than ``--kill-after=Ns``: busybox
#: ``timeout`` on Alpine images has the short options and not the long ones.
#: The ``command -v`` guard leaves an image without ``timeout`` running the CLI
#: unwrapped, which is how it behaved before, rather than failing every rollout
#: with exit 127.
DEADLINE_SHIM = """#!/bin/sh
# Injected by harbor_orchard: stop the CLI before Harbor stops waiting on it.
if [ -n "${{{deadline_var}:-}}" ] && command -v timeout >/dev/null 2>&1; then
    exec timeout -k {grace} "${deadline_var}" {binary} "$@"
fi
exec {binary} "$@"
"""


def deadline_shim(binary: str, *, deadline_var: str, grace: int) -> str:
    """Render :data:`DEADLINE_SHIM` for *binary*."""
    return DEADLINE_SHIM.format(
        deadline_var=deadline_var,
        grace=grace,
        binary=shlex.quote(binary),
    )


#: Where the captured patch is written. Under ``/logs/agent`` because Harbor
#: archives that directory to ``<trial>/agent/``, which is the exact layout
#: SWE-bench Pro V2's ``patch_replay:PatchReplayAgent`` globs for
#: (``<source_job>/instance_*/agent/model.patch``).
MODEL_PATCH_PATH = "/logs/agent/model.patch"


#: Snapshot the agent's work as a patch, for re-grading in a fresh sandbox.
#:
#: SWE-bench Pro V2's locked protocol grades the *patch*, not the container:
#: whatever the agent did to its own sandbox — a shimmed interpreter, an edited
#: ``node_modules``, a planted hook — is thrown away, and only this diff is
#: replayed into a clean image. Without the file the replay agent scores the
#: trial zero, so this runs for every agent and every dataset rather than being
#: wired to one benchmark. It is also the artifact a V1 run could not produce,
#: which is what made contamination auditing there a matter of reading
#: trajectories.
#:
#: The baseline is ``HEAD``, which *is* the base commit: these images are
#: checked out there and the agents here never commit. DeepSWE's collect hook
#: reads ``git diff <base> HEAD`` instead, and that spelling returns an empty
#: patch for a non-committing agent — the known cause of its zero-scoring
#: trials. ``git add -A`` before the diff is what picks up new and deleted
#: files, which a bare ``git diff`` would miss.
#:
#: ``git reset`` afterwards puts the index back, because on a shared-verifier
#: task the grader runs next *in this same container* and must see the tree the
#: agent left. ``-c safe.directory`` because this runs as root over a checkout
#: that may be owned by another user, where git would otherwise refuse and
#: leave an empty patch behind; git ignores the key on versions too old to know
#: it.
#:
#: Nothing here is allowed to fail the trial: every step swallows its status and
#: the routine always exits 0. A missing patch costs a re-grade, an agent phase
#: reported as failed costs the rollout.
MODEL_PATCH_ROUTINE = r"""
__hb_repo=""
for __hb_candidate in /app /testbed; do
    [ -d "$__hb_candidate" ] || continue
    __hb_repo=$(cd "$__hb_candidate" 2>/dev/null && \
        git rev-parse --show-toplevel 2>/dev/null) || __hb_repo=""
    [ -n "$__hb_repo" ] && break
done
if [ -z "$__hb_repo" ]; then
    echo "harbor-orchard: no git checkout under /app or /testbed; no patch captured" >&2
    exit 0
fi
mkdir -p "$(dirname {patch})" 2>/dev/null
cd "$__hb_repo" || exit 0
git -c safe.directory="$__hb_repo" add -A >/dev/null 2>&1
git -c safe.directory="$__hb_repo" diff --cached > {patch} 2>/dev/null
git -c safe.directory="$__hb_repo" reset -q >/dev/null 2>&1
echo "harbor-orchard: model.patch bytes: $(wc -c < {patch} 2>/dev/null || echo 0) ($__hb_repo)"
exit 0
"""


def model_patch_routine(patch_path: str = MODEL_PATCH_PATH) -> str:
    """Render :data:`MODEL_PATCH_ROUTINE` for *patch_path*."""
    return MODEL_PATCH_ROUTINE.format(patch=shlex.quote(patch_path))


#: What :func:`clean_tree_routine` prints, so the caller can parse one line
#: rather than guess from an exit status.
CLEAN_TREE_PREFIX = "harbor-orchard: tree "


#: Check that the agent is about to start on the checkout the image shipped.
#:
#: A trial is only a measurement if the agent starts where every other trial
#: starts. On the 2026-09-24 V2 sweep that stopped being true for 96 trials:
#: the sandbox client re-submits ``POST /exec`` when the HTTP call carrying the
#: result dies (``orchard_env/client/sandbox_client.py``, the bare ``except``
#: under "fall back to non-wait + polling"), the first agent is still running in
#: the pod, and the second one starts on top of a solution the first had already
#: written. Nothing upstream of it saw an error, so the run reported a score.
#:
#: Tracked files only. A clean V2 image legitimately carries untracked files —
#: upstream's own leak probe reports ``UNTRACKED=3`` on a pristine NodeBB image
#: and ``MODIFIED=0``, and this is the same measure, so the two agree on what
#: "clean" means.
#:
#: Like :data:`MODEL_PATCH_ROUTINE` this never fails on its own terms: it
#: reports and exits 0, and the caller decides whether a dirty tree ends the
#: trial. An image with no ``git`` and an image with no checkout both report
#: ``unknown`` rather than failing a rollout over a check.
CLEAN_TREE_ROUTINE = r"""
__hb_repo=""
for __hb_candidate in /app /testbed; do
    [ -d "$__hb_candidate" ] || continue
    __hb_repo=$(cd "$__hb_candidate" 2>/dev/null && \
        git rev-parse --show-toplevel 2>/dev/null) || __hb_repo=""
    [ -n "$__hb_repo" ] && break
done
if [ -z "$__hb_repo" ]; then
    echo "{prefix}unknown: no git checkout under /app or /testbed"
    exit 0
fi
cd "$__hb_repo" || exit 0
if ! __hb_dirty=$(git -c safe.directory="$__hb_repo" status --porcelain \
    --untracked-files=no 2>/dev/null); then
    echo "{prefix}unknown: git status failed in $__hb_repo"
    exit 0
fi
if [ -z "$__hb_dirty" ]; then
    echo "{prefix}clean ($__hb_repo)"
    exit 0
fi
__hb_count=$(printf '%s\n' "$__hb_dirty" | wc -l | tr -d ' ')
echo "{prefix}dirty ($__hb_repo): $__hb_count tracked file(s) already modified"
printf '%s\n' "$__hb_dirty" | head -20
exit 0
"""


def clean_tree_routine() -> str:
    """Render :data:`CLEAN_TREE_ROUTINE`."""
    return CLEAN_TREE_ROUTINE.format(prefix=CLEAN_TREE_PREFIX)


#: Make ``localhost`` resolve to 127.0.0.1 ahead of ``::1``.
#:
#: A benchmark's own test suite regularly starts a server and then talks to it
#: over ``http://localhost:<port>``. Bind the IPv4 wildcard, as NodeBB does —
#: "listening on 0.0.0.0:4568" — and that server is simply not there on ``::1``.
#: Upstream never meets this: Node asks ``getaddrinfo`` with ``AI_ADDRCONFIG``,
#: and a single-stack container has no global IPv6 address for glibc to keep
#: ``::1`` on the strength of. A dual-stack Kubernetes pod does, RFC 3484 sorts
#: it first, and the connection is refused before the server is consulted.
#:
#: Two mechanisms, because one alone does not cover the images here:
#:
#: * dropping the ``localhost`` alias from the ``::1`` line of ``/etc/hosts`` is
#:   what musl reads, and Alpine task images ignore ``gai.conf`` entirely;
#: * the ``gai.conf`` precedence rule is glibc's documented prefer-IPv4 knob,
#:   and covers names that are resolved rather than found in ``hosts``.
#:
#: Both are idempotent and both are best-effort: a read-only ``/etc`` leaves the
#: pod exactly as it was. ``cat >`` rather than ``sed -i`` because ``/etc/hosts``
#: is a bind mount, and an in-place edit that renames fails on one. The ``::1``
#: line keeps ``ip6-localhost`` and ``ip6-loopback``, so anything that asks for
#: IPv6 loopback *by that name* still gets it.
PREFER_IPV4_ROUTINE = r"""
set -u
__hb_changed=""
if [ -w /etc/hosts ] && awk '
    $1 == "::1" {
        out = $1
        for (i = 2; i <= NF; i++) {
            if ($i == "localhost") dropped = 1; else out = out " " $i
        }
        if (out != "::1") print out
        next
    }
    { print }
    END { exit dropped ? 0 : 1 }
' /etc/hosts > /tmp/.hb_hosts 2>/dev/null; then
    cat /tmp/.hb_hosts > /etc/hosts 2>/dev/null && __hb_changed="${__hb_changed}hosts "
fi
rm -f /tmp/.hb_hosts
if ! grep -q '^precedence ::ffff:0:0/96' /etc/gai.conf 2>/dev/null; then
    printf 'precedence ::ffff:0:0/96  100\n' >> /etc/gai.conf 2>/dev/null \
        && __hb_changed="${__hb_changed}gai.conf "
fi
echo "${__hb_changed:-nothing to change}"
exit 0
"""
