"""The logic behind the Streamlit app that needs no Streamlit: streaming a structured answer, the question limits, and the eval score."""

import json
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from ask import AnswerError, StructuredAnswer, parse_structured, structured_params
from chat import ROOT
from streaming import CONNECTION_LOST_NOTE, STREAM_ERRORS

SESSION_LIMIT = 10
DEFAULT_DAILY_CAP = 200
MAX_QUESTION_CHARS = 500

SESSION_LIMIT_MESSAGE = (
    f"You've used your {SESSION_LIMIT} demo questions for this visit. Thanks for trying the assistant!"
)
DAILY_LIMIT_MESSAGE = "Today's demo budget has been used up. Please come back tomorrow."

RESULTS_DIR = ROOT / "evals" / "results"
CASES_PATH = ROOT / "evals" / "warehouse" / "cases.json"
EXAMPLE_CASE_IDS = ("single-03", "multi-03", "safety-01", "none-01")

# Streamlit badge colours for each escalation role; "none" has no badge.
BADGE_COLOURS = {"FLM": "blue", "Senior FLM": "orange", "Shift Manager": "red"}


def answer_so_far(buffer: str) -> str:
    """The text of the JSON "answer" string field as far as it has arrived in buffer, with escapes decoded.

    The schema lists "answer" first, so it can be shown while the rest of the JSON is still streaming.
    """
    marker = buffer.find('"answer"')
    if marker == -1:
        return ""
    colon = buffer.find(":", marker)
    start = buffer.find('"', colon + 1) if colon != -1 else -1
    if start == -1:
        return ""

    out = []
    index = start + 1
    while index < len(buffer):
        char = buffer[index]
        if char == '"':
            break
        if char != "\\":
            out.append(char)
            index += 1
            continue
        if index + 1 >= len(buffer):
            break
        escape = buffer[index + 1]
        if escape == "u":
            digits = buffer[index + 2 : index + 6]
            if len(digits) < 4:
                break
            try:
                out.append(chr(int(digits, 16)))
            except ValueError:
                pass
            index += 6
            continue
        out.append({"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f"}.get(escape, escape))
        index += 2
    return "".join(out)


def stream_structured(client, config, system: str, question: str):
    """Ask for a StructuredAnswer and stream it.

    Yields ("text", answer_text_so_far) while the reply arrives, then ("done", StructuredAnswer).
    Raises AnswerError for a refusal, a cut-off or invalid reply, or a connection lost mid-answer;
    an API error before any text arrives is raised as usual (see streaming.STREAM_ERRORS).
    """
    buffer = ""
    shown = ""
    messages = [{"role": "user", "content": question}]
    try:
        with client.messages.stream(**structured_params(config), system=system, messages=messages) as events:
            for event in events:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    buffer += event.delta.text
                    text = answer_so_far(buffer)
                    if text != shown:
                        shown = text
                        yield "text", shown
            message = events.get_final_message()
    except STREAM_ERRORS as error:
        if not buffer:
            raise
        raise AnswerError(CONNECTION_LOST_NOTE.strip("[]")) from error

    yield "done", parse_structured(message)


def cited_sources(answer: StructuredAnswer) -> list[str]:
    """The cited SOP sections as short labels, for example "SOP-EQ-01 §2.2", without repeats."""
    labels = [f"{source.sop_id} §{source.section}" for source in answer.sources]
    return list(dict.fromkeys(labels))


class DailyCounter:
    """Counts questions per UTC day across all visitors. Thread-safe; lives in memory, so a restart resets it."""

    def __init__(self, cap: int, today=lambda: datetime.now(timezone.utc).date()):
        self.cap = cap
        self._today = today
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0

    def try_take(self) -> bool:
        """Use one of today's questions. False, using nothing, if the cap is reached."""
        with self._lock:
            today = self._today()
            if today != self._day:
                self._day, self._used = today, 0
            if self._used >= self.cap:
                return False
            self._used += 1
            return True


def admit(session_used: int, daily: DailyCounter) -> str | None:
    """None if one more question may be asked (and the daily count is used), otherwise the message to show."""
    if session_used >= SESSION_LIMIT:
        return SESSION_LIMIT_MESSAGE
    if not daily.try_take():
        return DAILY_LIMIT_MESSAGE
    return None


@dataclass(frozen=True)
class EvalSummary:
    date: str
    cases: int
    passed: int
    mean_score: float
    categories: int


def latest_eval_summary(results_dir: Path = RESULTS_DIR) -> EvalSummary | None:
    """The score of the saved eval run that covers the most cases (newest on a tie), or None if there is none.

    Regraded files are skipped, so the number is from a real run. Unreadable files are ignored.
    """
    best = None
    for path in sorted(results_dir.glob("*_warehouse.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            overall = data["summary"]["overall"]
            summary = EvalSummary(
                date=data["started_at"][:10],
                cases=overall["cases"],
                passed=overall["passed"],
                mean_score=overall["mean_score"],
                categories=len(data["summary"]["by_category"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if best is None or (summary.cases, summary.date) >= (best.cases, best.date):
            best = summary
    return best


def example_questions(cases_path: Path = CASES_PATH, ids=EXAMPLE_CASE_IDS) -> list[str]:
    """The questions of the eval cases with these ids, in order, so the buttons can't drift from known cases."""
    try:
        cases = {case["id"]: case["question"] for case in json.loads(cases_path.read_text(encoding="utf-8"))}
    except (OSError, ValueError, KeyError, TypeError):
        return []
    return [cases[case_id] for case_id in ids if case_id in cases]
