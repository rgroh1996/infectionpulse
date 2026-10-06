# A local morning brief with traceable evidence

`scripts/morning_brief.py` reads the same saved TabPFN predictions and policy as the dashboard and MCP service. It writes one short recommendation, the 0–100 **precaution index**, numerical forecast evidence, the activity inputs, and source/forecast dates. The authoritative JSON embeds the exact rendered text and forecast identifiers. It includes the service's probability readiness decision; it never invents personal infection odds.

The renderer is deterministic, not an LLM. It demonstrates the complete local data → forecast → contextual recommendation workflow without pretending that Hermes has been connected. No messages are sent and no scheduler is installed.

## Run locally

```sh
# Public, fictitious Berlin-to-Potsdam office/commute profile; tomorrow in Berlin.
.venv/bin/python scripts/morning_brief.py

# Choose a date that the saved forecast covers.
.venv/bin/python scripts/morning_brief.py --date 2026-09-22 --json

# Portable demonstration using actual recorded TabPFN predictions.
.venv/bin/python scripts/morning_brief.py --replay
```

The default outputs are `.runtime/morning-brief/latest.json` and `.txt`; replay writes `replay.json` and `.txt` separately and prominently labels itself **historical replay, not current advice**. Replay uses recorded source status and clock, and never pulls current disease context into the historical example. The `generated_at` field records actual execution time while `evidence_evaluated_at` records the replay clock.

The sample profile is fictional. Copy it to `.runtime/morning-profile.json` for personal edits and pass `--profile .runtime/morning-profile.json`. Keep personal files under ignored `.runtime/`; do not edit the tracked example with real personal details. Generated files have owner-only permissions. Concurrent writers to the same output are prevented with a local file lock; a separate replay can run while a live refresh is in progress. Outputs are replaced atomically. Use the JSON as the authoritative record if consuming both formats concurrently.

## Refresh and scheduling

```sh
.venv/bin/python scripts/morning_brief.py --refresh --profile .runtime/morning-profile.json
```

`--refresh` invokes `scripts/refresh.py` before constructing the brief. It checks the public ARE, COVID, influenza and RSV sources, uses the existing bounded Friday forecast schedule, and scores matured prospective forecasts. **Zero model-spending budget is the default.** Public source checks alone cannot generate a missing forecast: a new scheduled forecast requires an explicitly selected `--max-api-units` budget, or a prior approved manual `scripts/refresh.py --forecast-now --max-api-units BUDGET` run. Daily source checks can legitimately leave the same weekly forecast unchanged.

A refresh failure is disclosed in the brief. Saved predictions can still be used only when the normal service freshness and target-date checks pass. Missing, stale, past-date or unsupported forecasts produce a grey result with no score. The program never silently substitutes a replay. Exit codes: `0` usable brief, `1` refresh failed, `2` unavailable evidence or invalid arguments, `3` concurrent job skipped. Refresh progress goes to stderr so `--json` stdout remains parseable.

Example **to install later**, on a server whose cron implementation supports `CRON_TZ` (otherwise configure the scheduler's timezone explicitly):

```cron
CRON_TZ=Europe/Berlin
30 10 * * * cd /path/to/InfectionPulse && .venv/bin/python scripts/morning_brief.py --refresh --profile .runtime/morning-profile.json >> .runtime/morning-brief.log 2>&1
```

Create `.runtime/` before installing this entry. The 10:30 time permits the existing Friday forecast schedule (after 10:00 Berlin); a 07:00-only schedule would never trigger it. Configure a bounded API budget explicitly before enabling forecast generation. This command writes local artifacts only. It does not send email, contact Hermes, or enable model spending automatically.

## Optional Hermes connection, later

Hermes can read `latest.json` locally on its own server or call the existing read-only MCP tools described in [agent setup](agent-setup.md). Its role is wording and conversation, while the service owns the numbers and recommendation. A suggested instruction is:

> Summarize the supplied InfectionPulse brief in at most three sentences. Preserve its traffic light, activity date, precaution score label and recommendation. Include the median and 10th–90th percentile range, geography and observed-through date when status is available. Do not interpret the index as infection probability or infer a safer recommendation from missing data. If mode is historical_replay, lead with “Historical replay, not current advice.” If status is unavailable, say that current evidence is insufficient. Never add numerical claims or modify the activity assumptions.

Do not pass the TabPFN token to an agent. No Hermes invocation or delivery is claimed by this local demo. The generated JSON remains the auditable source even if a later LLM paraphrases it.

## Local validation recorded 21 September 2026

The real `--refresh` workflow completed at 07:37 UTC with public ARE, COVID, influenza and RSV source checks successful. No forecast inference was due on that Monday and the spending budget was zero. The live example for 22 September used the saved forecast issued 17 September, after confirming its source was still current: Eastern Germany median 7,881 ARE illnesses per 100,000, 10th–90th percentiles 5,964–9,887, observations through 13 September, forecast target 21–27 September. The fictional office/public-transit profile produced red, precaution index 85/100.

This was a **live source check with an existing weekly model forecast**, not a newly inferred forecast or replay. COVID observations advanced through 18 September; influenza and RSV source releases were unchanged. Before source access was allowed, the same command correctly produced grey, no score, and an explicit refresh-failed/source-check-stale explanation. These dated checks demonstrate operation, not future availability or predictive accuracy.
