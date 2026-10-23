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

## Decisions

### Thinking off by default (2026-10-06, to recheck in Stage 6)
`CLAUDE_THINKING` now defaults to `off` (was `on`), with effort `low` and
streaming on. Manual streaming test with `claude-sonnet-5-5`:

| Streaming | Thinking | Time to first text | Total time |
|---|---|---|---|
| off | on | 9.5s | — |
| on | on | 5.9s | — |
| on | off | 2.0s | 5.0s |

With thinking off, the answer was correct and used about half the output
tokens. Operatives read answers on the warehouse floor, so a first line in
about 2 seconds matters.

To recheck in Stage 6 with the 20-question test set:
- Answer quality with thinking off versus on, especially for questions that
  span more than one SOP. An earlier single comparison found the thinking-off
  answer slightly less precise.
- Time to first text, total time and output tokens across all 20 questions,
  not one.

`claude-opus-5-5` cannot turn thinking off, so `CLAUDE_MODEL=claude-opus-5-5`
needs `CLAUDE_THINKING=on`.

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

### JSON answer quality (from the same test)
All four findings below are fixed or resolved. Confirmed with real
`--json` runs on 2026-10-06: the damaged-pallet question returned
`safety_alert: false`, and the forklift injury question returned
`safety_alert: true` with `coverage: "partial"`.

- **Fixed — `safety_alert` too broad:** it was `true` for "What do I do with a
  damaged pallet?". The schema now says it is true only when a person is
  injured or at immediate risk and must stop and alert someone now, and that
  a damaged pallet taken out of use is not a safety alert.
- **Fixed — `covered_by_sops` couldn't say "partly":** the forklift injury
  question was marked `true`, but only the injury steps are covered. The
  field is now `coverage`, with the values `full`, `partial` and `none`.
- **Fixed — inconsistent section format:** `sources[].section` came back as
  both "Section 2" and "Section 1 (Reporting)". The schema now asks for a
  bare number ("4.2" for a step, "4" for a section), or the heading as
  written for parts without numbers, such as "Reporting / Escalation".
- **Resolved — hired-pallet rule applied to all pallets:** one answer said
  not to dismantle or dispose of any pallet on site. That answer was right;
  the SOP was unclear. No pallet, hired or owned, is dismantled or disposed of
  on site; all are collected. SOP-EQ-01 2.6 (version 1.2) now says so
  plainly. The system prompt also now tells Claude to apply a rule only to
  the items it covers.

### Eval grader has no "allowed extra citations" setting
`expected_sop_ids` means "these SOPs must all be cited", and extra citations
are tolerated. An empty list means "cite nothing". There is nothing in
between, such as "may cite these, but nothing else". So unrelated extra SOP
citations are caught only by the judge, not by the code checks. Seen in
`none-02` (2026-10-23), where citing SOP-MH-01 was fine, so the case now
requires only SOP-EQ-01.

### Intended behaviour (not an issue)
- **`escalate_to` is "FLM" for questions the SOPs don't cover.** The system
  prompt tells users to ask their FLM when the SOPs don't cover a question,
  so "FLM" is correct by design.
