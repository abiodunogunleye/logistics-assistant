# Logistics Assistant: Spec v0.1

## Problem
In small UK warehouses (10–100 staff), operatives and new starters ask supervisors
the same questions all day: how to handle damaged stock, where an item is, what the
procedure is. The answers sit in SOP documents nobody reads. Incidents get logged
late or inconsistently. Supervisors lose hours to interruptions.

## Users
- Warehouse operatives: ask questions, report incidents
- Shift supervisors: receive incident alerts, approve messages
- Warehouse manager (the buyer): wants fewer errors and faster onboarding

## The four jobs
1. Answer SOP questions, citing the source document
2. Check stock levels by SKU
3. Log incidents in a consistent format
4. Draft messages to supervisors, sent only after human approval

## Out of scope for v1
Connection to a real WMS, voice input, multiple sites, mobile app

## Success measures
- Correct answers on at least 18 of 20 test questions
- An incident logged in under 1 minute
- Every outgoing message approved by a human

## Tech
Python, Claude API, MCP server, Claude Agent SDK, GitHub

## Known issues

### Slow replies from `ask.py --json` (to measure in Stage 6)
Structured-output (JSON) replies were much slower than plain replies in the
first manual test on 2026-10-06 (`claude-sonnet-5-5`, thinking on, effort low):

| Question | Output tokens | Total time |
|---|---|---|
| What do I do with a damaged pallet? | 1,214 | 28.5s |
| Someone's been hit by a forklift. | 462 | 41.2s |
| What's the holiday booking process? | 109 | 102.2s |

Plain chat replies of a similar size took about 5s. The shortest JSON reply
was the slowest, so reply length doesn't explain it. The cause is not yet
known. These are single wall-clock timings that include Python start-up.

To measure:
- Time the API call inside `ask.py`, not the whole command.
- Run the same questions several times in plain and JSON mode, and compare.
- Run with `ANTHROPIC_LOG=info` to see whether the SDK is retrying requests.
- Check whether the first request with a new schema is slower than later
  ones (the API prepares each new schema once and reuses it for 24 hours).
