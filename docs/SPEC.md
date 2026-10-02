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
