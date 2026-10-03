"""Provider config staged into the pod so an in-pod agent CLI can reach a
self-hosted OpenAI-compatible server.

Codex reads ``OPENAI_BASE_URL`` only for its *built-in* provider, and that
provider talks to api.openai.com whatever the variable says. Reaching a
self-hosted server therefore requires a declared provider, which exists only in
the config file — without it every trial dies on turn 1 with a 401 from
api.openai.com and a non-zero exit, which reads as an agent failure.

This duplicates ``orchard_evalkit.harnesses.installed_cli`` on purpose: that
package drives its own rollouts and never has to be installed for a Harbor run,
and a Harbor run must not depend on it.
"""

from __future__ import annotations

import json

#: Codex declines to create its PATH helper binaries when CODEX_HOME sits under
#: the system temp dir, so this deliberately is not /tmp.
CODEX_HOME = "/var/tmp/orchard-codex"
CODEX_CONFIG_PATH = f"{CODEX_HOME}/config.toml"

#: Protocols a staged provider config may declare. "responses" is what current
#: codex expects; "chat" needs a codex predating codex#10157 but works against a
#: server exposing only /v1/chat/completions.
WIRE_APIS = ("responses", "chat")

#: The feature switches all exist for one reason: codex groups some built-in
#: tools into Responses API ``{"type": "namespace"}`` entries, which only
#: OpenAI's own endpoint accepts — vLLM and SGLang reject the whole request
#: body, so the very first turn fails. The reasoning-summary switches are the
#: same problem one turn later, when codex replays its own reasoning items.
CODEX_CONFIG_TEMPLATE = """\
model_provider = "orchard"
web_search = "disabled"
model_supports_reasoning_summaries = false
model_reasoning_summary = "none"

[model_providers.orchard]
name = "orchard"
base_url = {base_url}
env_key = {api_key_env}
wire_api = {wire_api}

[features]
multi_agent = false
apps = false
memories = false
remote_plugin = false

[agents]
enabled = false

[tools]
web_search = false
"""


def codex_config(
    base_url: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
    wire_api: str = "responses",
) -> str:
    """Render ``config.toml`` pointing codex at *base_url*."""
    return CODEX_CONFIG_TEMPLATE.format(
        base_url=json.dumps(base_url),
        api_key_env=json.dumps(api_key_env),
        wire_api=json.dumps(wire_api),
    )


#: Config directory for pi, overriding ~/.pi/agent so nothing depends on the
#: task image having a writable HOME.
PI_HOME = "/var/tmp/orchard-pi"
PI_CONFIG_PATH = f"{PI_HOME}/models.json"

#: What pi's model picker matches on. A served id is often a filesystem path,
#: and pi reads ``--model`` as an optional ``provider/id``, so the raw path is
#: not safe to pass on a command line.
PI_MODEL_ALIAS = "orchard-model"

#: pi validates ``--model`` against its own catalog and exits with "Model ... not
#: found" for anything absent from it, so a self-hosted endpoint has to be
#: declared as a custom provider. The compat flags are the ones pi documents for
#: vLLM/SGLang-class servers, which reject the ``developer`` role and
#: ``reasoning_effort`` pi otherwise sends to reasoning-capable models.
PI_CONFIG_TEMPLATE = """\
{{
  "providers": {{
    "orchard": {{
      "baseUrl": {base_url},
      "api": "openai-completions",
      "apiKey": {api_key_ref},
      "compat": {{
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      }},
      "models": [
        {{
          "id": {model},
          "name": {model_alias}{model_limits}
        }}
      ]
    }}
  }}
}}
"""


def pi_config(
    base_url: str,
    model: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
    model_alias: str = PI_MODEL_ALIAS,
    max_tokens: int | None = None,
) -> str:
    """Render ``models.json`` declaring *model* on *base_url* as a provider.

    *max_tokens* becomes pi's ``maxTokens`` for this model. Left ``None`` the
    field is omitted and pi applies its own default (16384) — the same value,
    but chosen by pi rather than recorded by the run, which is why callers
    normally pass it explicitly. See :attr:`OrchardSettings.pi_max_tokens` for
    why raising it is not the free win it looks like.
    """
    return PI_CONFIG_TEMPLATE.format(
        base_url=json.dumps(base_url),
        api_key_ref=json.dumps(f"${api_key_env}"),
        model=json.dumps(model),
        model_alias=json.dumps(model_alias),
        model_limits=(
            "" if max_tokens is None else f",\n          \"maxTokens\": {max_tokens:d}"
        ),
    )
