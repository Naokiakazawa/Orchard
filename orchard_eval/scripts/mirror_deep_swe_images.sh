#!/usr/bin/env bash
# Mirror every DeepSWE 1.1 task image from public.ecr.aws to a registry you
# control, so a 113-task run is not one anonymous pull quota away from failing.
#
# public.ecr.aws throttles anonymous pulls hard, and a DeepSWE run pulls each
# task image at least twice — once for the agent pod, once to build the
# verifier from tests/Dockerfile. Multiply by concurrency and the quota is the
# first thing that breaks, reported as a build failure on a random task.
#
# The image of record is `[environment] docker_image` in each task.toml; the
# verifier's tests/Dockerfile FROMs the same reference. Both are collected here
# and deduplicated, so nothing that a run resolves is left behind.
#
# Usage:
#   harbor download datacurve/deep-swe-1-1@latest -o ./datasets   # once
#   ./scripts/mirror_deep_swe_images.sh                       # mirror everything
#   ./scripts/mirror_deep_swe_images.sh ./datasets/deep-swe-1-1
#   DRY_RUN=1 ./scripts/mirror_deep_swe_images.sh             # print the plan only
#   MIRROR_MODE=imagetools ./scripts/mirror_deep_swe_images.sh # no local disk
#
# Environment:
#   DEST_REPO     destination repository (default wenlinyao/deep-swe). The
#                 source tag is reused verbatim, so tags stay recognizable.
#                 The provider rewrites the ECR repository onto this default;
#                 change it here and set ORCHARD_HARBOR_IMAGE_REMAP to match,
#                 e.g. public.ecr.aws/d3j8x8q7/swe-bench-202605=yourorg/deep-swe
#   TASKS_DIR     tasks root. Unset, the usual locations are searched; pass a
#                 path as $1 if the tree lives somewhere else. Any ancestor of
#                 the task directories works — task.toml is found at any depth.
#   MIRROR_MODE   pull    — docker pull / tag / push (default; what you tested)
#                 imagetools — `docker buildx imagetools create`, which copies
#                 registry-to-registry. No local layers, no disk, and multi-arch
#                 manifests survive; a plain pull flattens them to your arch.
#   PRUNE         1 (default in pull mode) removes the local copies after a
#                 successful push. 113 SWE images do not fit on most disks.
#   FORCE         1 re-pushes tags that already exist at the destination.
#   RETRIES       attempts per image (default 5), backing off 30s, 60s, 120s...
#                 which is what a rate limit actually needs.
#   MAP_FILE      where the source→destination map is written
#                 (default ./deep-swe-mirror-map.tsv)
#
# Log in first: `docker login` for Docker Hub. Anonymous pulls from
# public.ecr.aws are what you are escaping, but `aws ecr-public get-login-password`
# on the source side raises the quota too and costs nothing.
set -euo pipefail

TASKS_DIR="${1:-${TASKS_DIR:-}}"
DEST_REPO="${DEST_REPO:-wenlinyao/deep-swe}"
MIRROR_MODE="${MIRROR_MODE:-pull}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
RETRIES="${RETRIES:-5}"
MAP_FILE="${MAP_FILE:-./deep-swe-mirror-map.tsv}"

if [[ "$MIRROR_MODE" == "imagetools" ]]; then
  PRUNE="${PRUNE:-0}"
else
  PRUNE="${PRUNE:-1}"
fi

# --- locate the tasks -------------------------------------------------------
# $HOME is not always where the cache is — on a cluster box the login home and
# the writable home often differ — and the tree may have been unpacked by hand
# somewhere else entirely. So look, and say what was looked at.

has_tasks() {
  # Deep enough for the cache's content-addressable <org>/<name>/<digest>/ layout.
  [[ -d "$1" ]] && [[ -n "$(find "$1" -maxdepth 6 -name task.toml -print -quit 2>/dev/null)" ]]
}

CANDIDATES=()
if [[ -n "$TASKS_DIR" ]]; then
  CANDIDATES+=("$TASKS_DIR")
else
  CANDIDATES+=(
    "./datasets/deep-swe-1-1"
    "./deep-swe-1-1"
    "./tasks"
    "/tmp/deep-swe-audit/tasks"
    "${XDG_CACHE_HOME:-$HOME/.cache}/harbor"
    "$HOME/.cache/harbor"
    "/data/home/${USER}/.cache/harbor"
    "$(cd ~ 2>/dev/null && pwd)/.cache/harbor"
  )
fi

RESOLVED=""
for candidate in "${CANDIDATES[@]}"; do
  if has_tasks "$candidate"; then
    RESOLVED="$candidate"
    break
  fi
done

if [[ -z "$RESOLVED" ]]; then
  echo "error: no directory containing task.toml found. Tried:" >&2
  printf '  %s\n' "${CANDIDATES[@]}" >&2
  echo >&2
  echo "Pass the path as \$1, or download the dataset first:" >&2
  echo "  harbor download datacurve/deep-swe-1-1@latest -o ./datasets" >&2
  echo >&2
  echo "Already have it somewhere? Find it with:" >&2
  echo "  find ~ /tmp /data -maxdepth 8 -name task.toml 2>/dev/null | head" >&2
  exit 1
fi
TASKS_DIR="$RESOLVED"

case "$MIRROR_MODE" in
  pull | imagetools) ;;
  *)
    echo "error: MIRROR_MODE must be 'pull' or 'imagetools', got '$MIRROR_MODE'" >&2
    exit 1
    ;;
esac

# --- collect ---------------------------------------------------------------
# Both sources are read because a task is only mirrored if the *verifier* image
# is mirrored too, and nothing guarantees the Dockerfile FROM matches task.toml.

collect_images() {
  # task.toml: docker_image = "..." (schema 1.3) or image = "..." (older).
  find "$TASKS_DIR" -name task.toml -print0 |
    xargs -0 -r grep -hoE '^[[:space:]]*(docker_)?image[[:space:]]*=[[:space:]]*"[^"]+"' |
    sed -E 's/.*"([^"]+)".*/\1/'

  # Dockerfiles: the verifier base, and the environment base when built.
  find "$TASKS_DIR" -name Dockerfile -print0 |
    xargs -0 -r grep -hoiE '^[[:space:]]*FROM[[:space:]]+(--[a-z-]+=[^[:space:]]+[[:space:]]+)*[^[:space:]]+' |
    awk '{print $NF}'
}

mapfile -t IMAGES < <(
  collect_images |
    grep -vE '^\$|\$\{|^scratch$' |     # ARG-templated bases and scratch
    grep -E '[./]' |                     # a registry path, not a build stage name
    sort -u
)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
  echo "error: no images found under $TASKS_DIR" >&2
  exit 1
fi

TASK_COUNT="$(find "$TASKS_DIR" -name task.toml | wc -l)"
echo "Tasks root: ${TASKS_DIR}  (${TASK_COUNT} task.toml)"
echo "Found ${#IMAGES[@]} distinct image(s)"
echo "Destination: ${DEST_REPO}  (mode: ${MIRROR_MODE}, prune: ${PRUNE})"
echo

# --- destination naming ----------------------------------------------------
# The tag carries the task identity, so reusing it keeps `docker images` and
# any hand-written override readable. A digest reference has no tag to reuse;
# it becomes a sanitized one rather than silently colliding on :latest.

dest_for() {
  local src="$1" tag
  if [[ "$src" == *"@sha256:"* ]]; then
    tag="sha256-${src##*@sha256:}"
  else
    local last="${src##*/}"
    if [[ "$last" == *:* ]]; then
      tag="${last##*:}"
    else
      tag="latest"
    fi
  fi
  printf '%s:%s' "$DEST_REPO" "$tag"
}

# --- mirror ----------------------------------------------------------------

: >"$MAP_FILE"
failed=()
skipped=0
mirrored=0

# Backoff is the whole point: a rate limit is a wait, not an error, and
# retrying immediately just burns the next token bucket too.
retry() {
  local attempt=1 delay=30
  until "$@"; do
    if ((attempt >= RETRIES)); then
      return 1
    fi
    echo "  ...failed (attempt ${attempt}/${RETRIES}); retrying in ${delay}s" >&2
    sleep "$delay"
    attempt=$((attempt + 1))
    delay=$((delay * 2))
  done
  return 0
}

mirror_one() {
  local src="$1" dst="$2"

  if [[ "$MIRROR_MODE" == "imagetools" ]]; then
    retry docker buildx imagetools create --tag "$dst" "$src"
    return
  fi

  retry docker pull "$src" || return 1
  docker tag "$src" "$dst" || return 1
  retry docker push "$dst" || return 1

  if [[ "$PRUNE" == "1" ]]; then
    docker rmi "$dst" "$src" >/dev/null 2>&1 || true
  fi
  return 0
}

index=0
for src in "${IMAGES[@]}"; do
  index=$((index + 1))
  dst="$(dest_for "$src")"
  printf '%s\t%s\n' "$src" "$dst" >>"$MAP_FILE"

  echo "[${index}/${#IMAGES[@]}] ${src}"
  echo "        -> ${dst}"

  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi

  if [[ "$FORCE" != "1" ]] && docker manifest inspect "$dst" >/dev/null 2>&1; then
    echo "        already present, skipping"
    skipped=$((skipped + 1))
    continue
  fi

  if mirror_one "$src" "$dst"; then
    mirrored=$((mirrored + 1))
  else
    echo "        FAILED" >&2
    failed+=("$src")
  fi
done

echo
echo "Map written to ${MAP_FILE}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run: nothing pulled or pushed."
  exit 0
fi

echo "Mirrored ${mirrored}, skipped ${skipped}, failed ${#failed[@]} of ${#IMAGES[@]}"
if ((${#failed[@]} > 0)); then
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi
