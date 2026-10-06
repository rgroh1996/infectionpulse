# Connect a personal agent

The four InfectionPulse tools only read saved public forecasts. They cannot trigger
model inference, consume TabPFN tokens, schedule notifications, or send messages.
Your agent owns its personal memory, LLM account, and morning-brief schedule.

## Hermes on your server

Install this repository's service environment on the same machine as Hermes and
sync the public `models/service`, `data/service`, `data/respiratory/source_status.json`
and district catalog files. Keep the service's forecast and source refresh job running.
Add to `~/.hermes/config.yaml`, replacing paths with absolute server paths:

```yaml
mcp_servers:
  infectionpulse:
    command: /absolute/path/InfectionPulse/.venv/bin/python
    args:
      - /absolute/path/InfectionPulse/scripts/serve.py
      - --transport
      - stdio
```

For a service on a different machine, use the remote HTTP configuration supported by
your Hermes version, endpoint `https://YOUR_HOST/mcp/`, and an Authorization bearer
header with **INFECTIONPULSE_ACCESS_TOKEN**, not TABPFN_TOKEN. Prefer an SSH tunnel
if you do not already have HTTPS and authentication configured.
For a non-local HTTP hostname, set `INFECTIONPULSE_ALLOWED_HOSTS` on the service
to that hostname (and explicit port where used). Browser clients may also need
`INFECTIONPULSE_ALLOWED_ORIGINS`. Keep MCP host protection enabled.

Official instructions: https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp

## OpenClaw

Configure an outbound MCP server in `mcp.servers` using the same stdio command or
Streamable HTTP endpoint. The exact configuration is version-dependent; follow
https://docs.openclaw.ai/gateway/config-extensions and validate with
`openclaw mcp probe`. `openclaw mcp serve` exposes OpenClaw itself and is not the
client connection to InfectionPulse. No custom agent plugin is required.

## Agent instructions

1. Resolve home and destination names with `resolve_location`. Ask when ambiguous.
2. Check `get_source_status` and supported forecast dates.
3. Request only missing activity context. Pass district IDs, date, setting, duration,
   crowding, ventilation, and commute to `assess_activity`.
4. Give its recommendation in one sentence, including the activity date. Show the
   underlying forecast and source dates only when useful or requested.
5. Preserve grey/unknown and the stated macroregion. Never invent a personal
   infection percentage or describe green as a guarantee of safety.
6. Do not substitute next week's prediction for an unsupported activity tonight.
7. Use `explanation.summary` when asked why: it contains the actual forecast,
   threshold comparison, and activity assumptions. `explanation.alternatives`
   contains recomputed policy scenarios, including unchanged colours. These are
   decision-rule explanations, not SHAP or causal risk reductions. Quote a
   community probability only when `explanation.probability.status` is `available`;
   preserve its event definition and separate regional values. Never replace a
   withheld value with incidence or a personal percentage.
8. `precaution_score.value` is a coarse 0–100 policy score in **points**, not a
   percentage. Preserve null when evidence is unavailable. `disease_context`
   contains state-level observed influenza and RSV; `local_context` contains
   district COVID observations. Keep dates and geography attached, and never
   describe these observations as disease-specific forecasts or add them to ARE.

Example request: “I live in Berlin, work in Potsdam, take a crowded train for
35 minutes, and share an office for six hours next Tuesday. What precautions
make sense?” The agent must call tools and preserve the returned target dates.

## Validation status

REST, MCP stdio, and Streamable HTTP are tested locally using the actual MCP Python
SDK. Tests cover discovery, calls, schema parity, authentication, ambiguity, stale
sources, and unavailable target weeks. No agent-specific execution is claimed:
Hermes is available on the user's server and its live test is deferred by request.
OpenClaw is not installed in this workspace.
