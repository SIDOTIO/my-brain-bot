import os
import json
import logging
import uuid
from datetime import date, datetime, timedelta, time as dtime, timezone
from pathlib import Path
from collections import deque

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)
import openai
import anthropic

# Optional: Google Calendar
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    GCAL_AVAILABLE = True
except ImportError:
    GCAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WHISPER_KEY    = os.environ["WHISPER_KEY"]
ANTHROPIC_KEY  = os.environ["ANTHROPIC_KEY"]

openai.api_key = WHISPER_KEY
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

VAULT        = Path(__file__).parent / "vault"
CHAT_ID_FILE = VAULT / "chat_id.txt"
HISTORY_FILE = VAULT / "conversation_history.json"
TODAY        = lambda: date.today().isoformat()
NOW          = lambda: datetime.now().strftime("%Y-%m-%d %H:%M")

# User timezone offset (EST = UTC-5, EDT = UTC-4)
USER_TZ_NAME = "America/New_York"

# ---------------------------------------------------------------------------
# Conversation Memory — the core fix
# ---------------------------------------------------------------------------

MAX_HISTORY = 30  # Keep last 30 message pairs

def _load_history() -> list[dict]:
    """Load conversation history from disk."""
    VAULT.mkdir(parents=True, exist_ok=True)  # Ensure vault exists
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data[-MAX_HISTORY * 2:]  # Keep last N exchanges
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("Could not load history: %s", e)
            pass
    return []


def _save_history(messages: list[dict]) -> None:
    """Persist conversation history to disk."""
    VAULT.mkdir(parents=True, exist_ok=True)
    # Only keep the last MAX_HISTORY exchanges (user + assistant pairs)
    trimmed = messages[-MAX_HISTORY * 2:]
    HISTORY_FILE.write_text(
        json.dumps(trimmed, ensure_ascii=False, default=str),
        encoding="utf-8"
    )


def _add_to_history(role: str, content: str) -> None:
    """Add a single message to history."""
    history = _load_history()
    history.append({"role": role, "content": content})
    _save_history(history)


def _get_context_messages() -> list[dict]:
    """Get conversation history formatted for Claude API.

    Filters out tool_use/tool_result blocks and only keeps clean
    user/assistant text pairs for context.
    """
    history = _load_history()
    # Only include clean text messages (not tool results)
    clean = []
    for msg in history:
        if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
            clean.append({"role": msg["role"], "content": msg["content"]})
    return clean[-MAX_HISTORY * 2:]


# ---------------------------------------------------------------------------
# Google Calendar Integration
# ---------------------------------------------------------------------------

GCAL_CREDS_FILE = VAULT / "google_credentials.json"
GCAL_TOKEN_FILE = VAULT / "google_token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def _get_calendar_service():
    """Get authenticated Google Calendar service."""
    if not GCAL_AVAILABLE:
        return None
    if not GCAL_TOKEN_FILE.exists():
        return None
    try:
        creds = Credentials.from_authorized_user_file(str(GCAL_TOKEN_FILE), SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            GCAL_TOKEN_FILE.write_text(creds.to_json())
        if creds and creds.valid:
            return build("calendar", "v3", credentials=creds)
    except Exception as e:
        logger.error("Google Calendar auth failed: %s", e)
    return None


def gcal_get_events(date_str: str = None, days: int = 1) -> str:
    """Get calendar events for a date range."""
    service = _get_calendar_service()
    if not service:
        return "Google Calendar not connected. Run setup_gcal.py first."

    try:
        if date_str:
            start_date = datetime.fromisoformat(date_str)
        else:
            start_date = datetime.now(timezone.utc)

        time_min = start_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        time_max = (start_date + timedelta(days=days)).replace(hour=23, minute=59, second=59, microsecond=0).isoformat()

        events_result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = events_result.get("items", [])
        if not events:
            return f"No events found for {start_date.strftime('%B %d, %Y')}."

        lines = []
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "Untitled")
            lines.append(f"• {start}: {summary}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("Calendar read error: %s", e)
        return f"Error reading calendar: {e}"


def gcal_create_event(summary: str, start_time: str, end_time: str, description: str = "") -> str:
    """Create a Google Calendar event."""
    service = _get_calendar_service()
    if not service:
        return "Google Calendar not connected. Run setup_gcal.py first."

    try:
        event = {
            "summary": summary,
            "start": {"dateTime": start_time, "timeZone": USER_TZ_NAME},
            "end": {"dateTime": end_time, "timeZone": USER_TZ_NAME},
        }
        if description:
            event["description"] = description

        created = service.events().insert(calendarId="primary", body=event).execute()
        return f"Created event: {summary} — {created.get('htmlLink', 'done')}"
    except Exception as e:
        logger.error("Calendar create error: %s", e)
        return f"Error creating event: {e}"


def gcal_delete_event(event_id: str) -> str:
    """Delete a Google Calendar event by ID."""
    service = _get_calendar_service()
    if not service:
        return "Google Calendar not connected."
    try:
        service.events().delete(calendarId="primary", eventId=event_id).execute()
        return "Event deleted."
    except Exception as e:
        return f"Error deleting event: {e}"


def gcal_create_schedule(schedule_blocks: list[dict]) -> str:
    """Create multiple calendar events from a schedule. Each block: {summary, start, end}"""
    service = _get_calendar_service()
    if not service:
        return "Google Calendar not connected. Run setup_gcal.py first."

    created = []
    errors = []
    for block in schedule_blocks:
        try:
            event = {
                "summary": block["summary"],
                "start": {"dateTime": block["start"], "timeZone": USER_TZ_NAME},
                "end": {"dateTime": block["end"], "timeZone": USER_TZ_NAME},
            }
            if block.get("description"):
                event["description"] = block["description"]
            service.events().insert(calendarId="primary", body=event).execute()
            created.append(block["summary"])
        except Exception as e:
            errors.append(f"{block.get('summary', '?')}: {e}")

    result = f"Created {len(created)} events on your calendar."
    if errors:
        result += f"\n{len(errors)} failed: " + "; ".join(errors)
    return result


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _call_claude(system: str, user: str, max_tokens: int = 512) -> str:
    response = claude.messages.create(
        model="claude-haiku-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def _append(path: Path, text: str, entry_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n## {entry_date}\n\n{text}\n")


def _read_recent(folder: Path, today: str) -> str:
    today_file = folder / f"{today}.md"
    if today_file.exists():
        return today_file.read_text(encoding="utf-8").strip()
    files = sorted(folder.glob("*.md"), reverse=True)
    return files[0].read_text(encoding="utf-8").strip() if files else ""


def _read_days(folder: Path, days: int = 7) -> str:
    if not folder.exists():
        return ""
    cutoff = date.today() - timedelta(days=days)
    parts = []
    for f in sorted(folder.glob("*.md"), reverse=True):
        try:
            file_date = date.fromisoformat(f.stem)
        except ValueError:
            continue
        if file_date >= cutoff:
            parts.append(f.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def _save_chat_id(chat_id: int) -> None:
    VAULT.mkdir(parents=True, exist_ok=True)
    CHAT_ID_FILE.write_text(str(chat_id))


def _load_chat_id() -> int | None:
    if CHAT_ID_FILE.exists():
        try:
            return int(CHAT_ID_FILE.read_text().strip())
        except ValueError:
            return None
    return None


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return default


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(file_path: str) -> str:
    size = os.path.getsize(file_path)
    logger.info("Transcribing %s — file size: %d bytes", file_path, size)
    if size == 0:
        raise ValueError("Audio file is empty — download failed")
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    result = openai.audio.transcriptions.create(
        model="whisper-1",
        file=("voice.ogg", audio_bytes, "audio/ogg"),
    )
    return result.text


# ---------------------------------------------------------------------------
# Save agents (no LLM — zero cost)
# ---------------------------------------------------------------------------

def finance_agent(text: str) -> str:
    entry_date = TODAY()
    lower = text.lower()
    if any(w in lower for w in ("earned", "income", "received", "salary", "paid me")):
        target = VAULT / "income" / f"{entry_date}.md"
        category = "income"
    else:
        target = VAULT / "expenses" / f"{entry_date}.md"
        category = "expenses"
    _append(target, text, entry_date)
    return f"Saved to {category}."


def task_agent(text: str) -> str:
    entry_date = TODAY()
    lower = text.lower()
    if any(w in lower for w in ("school", "class", "homework", "assignment", "study")):
        target = VAULT / "todos-school" / f"{entry_date}.md"
        folder = "todos-school"
    else:
        target = VAULT / "todos-work" / f"{entry_date}.md"
        folder = "todos-work"
    _append(target, text, entry_date)
    return f"Saved to {folder}."


def scheduler_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "habits" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return "Saved schedule/habit."


def ideas_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "ideas" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return "Saved idea."


def journal_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "journal" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return "Saved journal entry."


# ---------------------------------------------------------------------------
# Command agents (LLM-powered)
# ---------------------------------------------------------------------------

def focus_agent() -> str:
    today = TODAY()
    parts = []
    profile_path = VAULT / "profile.md"
    if profile_path.exists():
        profile = profile_path.read_text(encoding="utf-8").strip()
        if profile:
            parts.append(f"Profile:\n{profile}")
    work = _read_recent(VAULT / "todos-work", today)
    if work:
        parts.append(f"Work todos:\n{work}")
    school = _read_recent(VAULT / "todos-school", today)
    if school:
        parts.append(f"School todos:\n{school}")
    if not parts:
        return "No todos or profile found yet. Add some tasks first!"
    return _call_claude(
        system=(
            "You are a personal assistant. Given the user's profile and todos, "
            "give a concise prioritized focus list for today. "
            "Be direct and brief — 5-8 bullet points max."
        ),
        user="\n\n".join(parts),
        max_tokens=512,
    )


def learn_agent(text: str) -> str:
    index_path = VAULT / "knowledge" / "index.json"
    items = _load_json(index_path, [])
    today = date.today()
    items.append({
        "id": str(uuid.uuid4())[:8],
        "content": text,
        "created": today.isoformat(),
        "next_review": (today + timedelta(days=1)).isoformat(),
        "interval": 1,
        "reviews": 0,
    })
    _save_json(index_path, items)
    return "Saved to knowledge. First review tomorrow."


FOLDER_KEYWORDS = {
    "expenses":    {"expense", "spent", "spending", "money", "cost", "bought", "paid", "bill", "purchase"},
    "income":      {"income", "earned", "salary", "received", "revenue"},
    "todos-work":  {"todo", "task", "work", "meeting", "deadline", "project"},
    "todos-school":{"school", "homework", "class", "assignment", "study"},
    "journal":     {"journal", "day", "entry", "wrote", "logged"},
    "habits":      {"habit", "routine", "streak", "morning", "evening"},
    "ideas":       {"idea", "concept", "brainstorm"},
    "decisions":   {"decision", "decided", "chose", "choice"},
    "body":        {"sleep", "slept", "energy", "workout", "gym", "body"},
    "reading-log": {"read", "reading", "book", "article"},
}


def recall_agent(query: str) -> str:
    query_lower = query.lower()
    matches = []
    targeted_folders = {
        folder for folder, keywords in FOLDER_KEYWORDS.items()
        if any(kw in query_lower for kw in keywords)
    }
    all_folders = [
        "journal", "ideas", "todos-work", "todos-school",
        "habits", "expenses", "income", "reading-log", "decisions", "body",
    ]
    search_folders = targeted_folders if targeted_folders else all_folders
    for folder in search_folders:
        folder_path = VAULT / folder
        if not folder_path.exists():
            continue
        for f in sorted(folder_path.glob("*.md"), reverse=True)[:14]:
            content = f.read_text(encoding="utf-8").strip()
            if content:
                matches.append(f"[{folder}/{f.name}]\n{content[:600]}")
    if not targeted_folders:
        matches = [m for m in matches if query_lower in m.lower()]
    index_path = VAULT / "knowledge" / "index.json"
    for item in _load_json(index_path, []):
        if query_lower in item.get("content", "").lower():
            matches.append(f"[knowledge]\n{item['content']}")
    if not matches:
        return f"Nothing found in your vault for '{query}'."
    combined = "\n\n---\n\n".join(matches[:10])
    return _call_claude(
        system=(
            "You are a personal knowledge assistant. Given vault entries, "
            "synthesize a clear, useful answer to the user's query. Be concise."
        ),
        user=f"Query: {query}\n\nVault entries:\n{combined}",
        max_tokens=600,
    )


def read_agent(text: str) -> str:
    entry_date = TODAY()
    insight = _call_claude(
        system="Extract the single most important insight from this reading note in one sentence.",
        user=text,
        max_tokens=100,
    )
    target = VAULT / "reading-log" / f"{entry_date}.md"
    _append(target, f"{text}\n\n**Key insight:** {insight}", entry_date)
    return f"Saved to reading-log.\nKey insight: {insight}"


def decide_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "decisions" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return "Decision logged. I'll resurface this in a month for outcome review."


def body_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "body" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Body log saved for {entry_date}."


def develop_agent(idea_name: str) -> str:
    idea_content = ""
    ideas_folder = VAULT / "ideas"
    for f in sorted(ideas_folder.glob("*.md"), reverse=True) if ideas_folder.exists() else []:
        content = f.read_text(encoding="utf-8")
        if idea_name.lower() in content.lower():
            idea_content = content[:800]
            break
    if not idea_content:
        idea_content = f"Idea: {idea_name} (no notes found yet)"
    questions = _call_claude(
        system=(
            "You are a Socratic thinking partner. Given an idea, generate exactly 3 sharp questions "
            "that help the thinker pressure-test assumptions, identify the real problem, and clarify "
            "next steps. Format as a numbered list."
        ),
        user=idea_content,
        max_tokens=300,
    )
    maturity_path = VAULT / "ideas" / "maturity.json"
    maturity = _load_json(maturity_path, {})
    key = idea_name.lower().strip()
    maturity[key] = maturity.get(key, 0) + 1
    _save_json(maturity_path, maturity)
    score = maturity[key]
    flag = " — SERIOUS IDEA, consider making this a project!" if score >= 3 else ""
    return (
        f"Developing: {idea_name} (maturity: {score}/3{flag})\n\n"
        f"{questions}\n\n"
        "Voice or type your thoughts — they'll be saved to your journal."
    )


def week_agent() -> str:
    parts = []
    for label, folder, limit in [
        ("Journal entries", "journal", 1500),
        ("Work todos", "todos-work", 800),
        ("Habits", "habits", 800),
        ("Expenses", "expenses", 500),
        ("Ideas", "ideas", 500),
    ]:
        content = _read_days(VAULT / folder, 7)
        if content:
            parts.append(f"{label}:\n{content[:limit]}")
    if not parts:
        return "Not enough data for a weekly review yet. Keep logging!"
    return _call_claude(
        system=(
            "You are a personal analyst. Generate a weekly review covering: "
            "1) Spending patterns and anything unusual, "
            "2) Habits — which were consistent vs skipped, and on what days, "
            "3) Recurring themes in journal that deserve attention, "
            "4) Ideas that connect to each other. "
            "Be honest and specific. Max 300 words."
        ),
        user="\n\n".join(parts),
        max_tokens=600,
    )


# ---------------------------------------------------------------------------
# Scheduled jobs
# ---------------------------------------------------------------------------

def _get_due_reviews() -> list[str]:
    index_path = VAULT / "knowledge" / "index.json"
    today = date.today()
    due = []
    for item in _load_json(index_path, []):
        try:
            if date.fromisoformat(item["next_review"]) <= today:
                due.append(item["content"])
        except (KeyError, ValueError):
            continue
    return due[:3]


async def morning_briefing(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _load_chat_id()
    if not chat_id:
        return

    today = TODAY()
    parts = []

    profile_path = VAULT / "profile.md"
    if profile_path.exists():
        p = profile_path.read_text(encoding="utf-8").strip()
        if p:
            parts.append(f"Profile:\n{p}")

    for label, folder in [("Work todos", "todos-work"), ("School todos", "todos-school")]:
        content = _read_recent(VAULT / folder, today)
        if content:
            parts.append(f"{label}:\n{content}")

    habits = _read_recent(VAULT / "habits", today)
    if habits:
        parts.append(f"Habits:\n{habits[:300]}")

    journal_snippets = []
    jfolder = VAULT / "journal"
    if jfolder.exists():
        for f in sorted(jfolder.glob("*.md"), reverse=True)[:3]:
            journal_snippets.append(f.read_text(encoding="utf-8").strip()[:200])
    if journal_snippets:
        parts.append("Recent journal:\n" + "\n---\n".join(journal_snippets))

    rlfolder = VAULT / "reading-log"
    if rlfolder.exists():
        reading_files = sorted(rlfolder.glob("*.md"), reverse=True)
        if reading_files:
            parts.append(f"Recent reading:\n{reading_files[0].read_text(encoding='utf-8').strip()[:300]}")

    # Include today's calendar if available
    cal_events = gcal_get_events()
    if "not connected" not in cal_events.lower() and "no events" not in cal_events.lower():
        parts.append(f"Today's calendar:\n{cal_events}")

    brief = _call_claude(
        system=(
            "You are the 60-year-old version of Sid P — the version that achieved everything. "
            "Give him his morning intelligence brief. Be direct, motivating, no fluff. Include: "
            "1) Top 3 priorities for today (numbered), "
            "2) One habit to focus on, "
            "3) One connection between recent reading/learning and current work, "
            "4) If there are calendar events, remind him what's on deck. "
            "Max 200 words. Talk like someone who loves him but won't let him coast."
        ),
        user="\n\n".join(parts) if parts else "No context yet — tell him to start logging.",
        max_tokens=400,
    )

    message = f"Good morning! Your daily brief:\n\n{brief}"

    due = _get_due_reviews()
    if due:
        message += "\n\nReview time (spaced repetition):\n" + "\n".join(f"• {r}" for r in due)

    await context.bot.send_message(chat_id=chat_id, text=message)


async def evening_checks(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = _load_chat_id()
    if not chat_id:
        return

    today = date.today()

    # Sunday: weekly review
    if today.weekday() == 6:
        review = week_agent()
        await context.bot.send_message(chat_id=chat_id, text=f"Weekly review:\n\n{review}")

    # 1st of month: financial trajectory
    if today.day == 1:
        expenses = _read_days(VAULT / "expenses", 31)
        income   = _read_days(VAULT / "income",   31)
        if expenses or income:
            fin = _call_claude(
                system=(
                    "You are a personal finance advisor. Analyze the user's last month of expenses and income. "
                    "Calculate: total spent by rough category, savings rate, and project their trajectory. "
                    "Give a plain English verdict: on track, behind, or ahead? Be honest. Max 200 words."
                ),
                user=f"Expenses:\n{expenses or 'None'}\n\nIncome:\n{income or 'None'}",
                max_tokens=400,
            )
            await context.bot.send_message(chat_id=chat_id, text=f"Monthly financial review:\n\n{fin}")

    # Every 14 days: auto-update profile
    last_update_path = VAULT / "profile_last_update.txt"
    should_update = True
    if last_update_path.exists():
        try:
            last = date.fromisoformat(last_update_path.read_text().strip())
            should_update = (today - last).days >= 14
        except ValueError:
            pass

    if should_update:
        journal = _read_days(VAULT / "journal", 14)
        todos   = _read_days(VAULT / "todos-work", 14)
        habits  = _read_days(VAULT / "habits", 14)
        ideas   = _read_days(VAULT / "ideas", 14)
        if journal or todos or habits:
            profile_path = VAULT / "profile.md"
            current = profile_path.read_text(encoding="utf-8").strip() if profile_path.exists() else ""
            updated = _call_claude(
                system=(
                    "You are a behavioral analyst. Based on 14 days of the user's activity, update their profile. "
                    "IMPORTANT: Keep the existing profile structure and all permanent info (identity, background, etc). "
                    "Only update the behavioral observations section. Add: observed strengths, current priorities "
                    "(based on behavior, not stated goals), recurring patterns, and contradictions. "
                    "Be honest and specific. Write in second person. Max 300 words for the update section."
                ),
                user=(
                    f"Current profile:\n{current or 'None'}\n\n"
                    f"Journal (14 days):\n{journal[:1000]}\n\n"
                    f"Todos:\n{todos[:500]}\n\n"
                    f"Habits:\n{habits[:500]}\n\n"
                    f"Ideas:\n{ideas[:300]}"
                ),
                max_tokens=600,
            )
            profile_path.write_text(updated, encoding="utf-8")
            last_update_path.write_text(today.isoformat())
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"Profile updated from your last 14 days:\n\n{updated[:500]}"
            )


# ---------------------------------------------------------------------------
# AI Orchestrator — Claude Sonnet with full conversation memory
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "save_expense",
        "description": "Save a financial expense or income entry",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "category": {"type": "string", "enum": ["expense", "income"]},
            },
            "required": ["text", "category"],
        },
    },
    {
        "name": "save_todo",
        "description": "Save a task or to-do item",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "context": {"type": "string", "enum": ["work", "school"]},
            },
            "required": ["text", "context"],
        },
    },
    {
        "name": "save_journal",
        "description": "Save a personal journal entry, feeling, reflection, or daily log",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_idea",
        "description": "Save an idea, concept, or creative thought",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_habit",
        "description": "Save a habit, routine, or schedule entry",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_knowledge",
        "description": "Save something the user wants to learn and retain, with spaced repetition scheduling",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_decision",
        "description": "Save a decision the user has made with their reasoning",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_body",
        "description": "Save health and body data: sleep hours, energy level, workout",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_reading",
        "description": "Save a reading note from a book or article",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "search_vault",
        "description": "Search the user's saved notes to answer a question about past data",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for"},
                "folders": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "journal", "ideas", "todos-work", "todos-school",
                            "habits", "expenses", "income", "reading-log",
                            "decisions", "body", "knowledge",
                        ],
                    },
                    "description": "Folders to search",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_focus_list",
        "description": "Get the user's current todos and priorities to build a focus plan for today",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_weekly_review",
        "description": "Generate a weekly review of activities, habits, spending, and ideas",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_calendar_events",
        "description": "Get events from Google Calendar for a specific date or range",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "ISO date like 2026-03-22. Defaults to today."},
                "days": {"type": "integer", "description": "Number of days to look ahead. Default 1."},
            },
            "required": [],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create an event on Google Calendar",
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Event title"},
                "start_time": {"type": "string", "description": "ISO datetime like 2026-03-22T10:00:00"},
                "end_time": {"type": "string", "description": "ISO datetime like 2026-03-22T11:30:00"},
                "description": {"type": "string", "description": "Optional event description"},
            },
            "required": ["summary", "start_time", "end_time"],
        },
    },
    {
        "name": "create_full_schedule",
        "description": "Create a full day schedule on Google Calendar from multiple time blocks. Use this when the user asks you to build a schedule and put it on their calendar.",
        "input_schema": {
            "type": "object",
            "properties": {
                "blocks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "start": {"type": "string", "description": "ISO datetime"},
                            "end": {"type": "string", "description": "ISO datetime"},
                            "description": {"type": "string"},
                        },
                        "required": ["summary", "start", "end"],
                    },
                    "description": "List of schedule blocks to create as calendar events",
                },
            },
            "required": ["blocks"],
        },
    },
]


def _execute_tool(name: str, inputs: dict) -> str:
    entry_date = TODAY()

    if name == "save_expense":
        folder = "income" if inputs.get("category") == "income" else "expenses"
        _append(VAULT / folder / f"{entry_date}.md", inputs["text"], entry_date)
        return f"Saved to {folder}."

    if name == "save_todo":
        folder = "todos-school" if inputs.get("context") == "school" else "todos-work"
        _append(VAULT / folder / f"{entry_date}.md", inputs["text"], entry_date)
        return f"Saved to {folder}."

    if name == "save_journal":
        _append(VAULT / "journal" / f"{entry_date}.md", inputs["text"], entry_date)
        return "Saved journal entry."

    if name == "save_idea":
        _append(VAULT / "ideas" / f"{entry_date}.md", inputs["text"], entry_date)
        return "Saved idea."

    if name == "save_habit":
        _append(VAULT / "habits" / f"{entry_date}.md", inputs["text"], entry_date)
        return "Saved habit/schedule."

    if name == "save_knowledge":
        return learn_agent(inputs["text"])

    if name == "save_decision":
        _append(VAULT / "decisions" / f"{entry_date}.md", inputs["text"], entry_date)
        return "Saved decision."

    if name == "save_body":
        _append(VAULT / "body" / f"{entry_date}.md", inputs["text"], entry_date)
        return "Saved body data."

    if name == "save_reading":
        return read_agent(inputs["text"])

    if name == "search_vault":
        query = inputs["query"]
        folders = inputs.get("folders") or [
            "journal", "ideas", "todos-work", "todos-school",
            "habits", "expenses", "income", "reading-log", "decisions", "body",
        ]
        results = []
        for folder in folders:
            folder_path = VAULT / folder
            if not folder_path.exists():
                continue
            for f in sorted(folder_path.glob("*.md"), reverse=True)[:14]:
                content = f.read_text(encoding="utf-8").strip()
                if content:
                    results.append(f"[{folder}/{f.name}]\n{content[:600]}")
        for item in _load_json(VAULT / "knowledge" / "index.json", []):
            results.append(f"[knowledge]\n{item['content']}")
        return "\n\n---\n\n".join(results[:15]) if results else "No entries found."

    if name == "get_focus_list":
        today = TODAY()
        parts = []
        profile_path = VAULT / "profile.md"
        if profile_path.exists():
            parts.append(f"Profile:\n{profile_path.read_text(encoding='utf-8').strip()[:500]}")
        work = _read_recent(VAULT / "todos-work", today)
        if work:
            parts.append(f"Work todos:\n{work}")
        school = _read_recent(VAULT / "todos-school", today)
        if school:
            parts.append(f"School todos:\n{school}")
        return "\n\n".join(parts) or "No todos found yet."

    if name == "get_weekly_review":
        return week_agent()

    if name == "get_calendar_events":
        return gcal_get_events(
            date_str=inputs.get("date"),
            days=inputs.get("days", 1)
        )

    if name == "create_calendar_event":
        return gcal_create_event(
            summary=inputs["summary"],
            start_time=inputs["start_time"],
            end_time=inputs["end_time"],
            description=inputs.get("description", ""),
        )

    if name == "create_full_schedule":
        return gcal_create_schedule(inputs.get("blocks", []))

    return "Unknown tool."


def ai_orchestrate(user_text: str) -> str:
    """Main AI orchestrator with FULL conversation memory."""

    # Load full profile
    profile = ""
    profile_path = VAULT / "profile.md"
    if profile_path.exists():
        profile = profile_path.read_text(encoding="utf-8").strip()

    # Load today's context
    today = TODAY()
    today_context = []

    # Recent journal
    journal = _read_recent(VAULT / "journal", today)
    if journal:
        today_context.append(f"Today's journal:\n{journal[:500]}")

    # Recent todos
    for label, folder in [("Work todos", "todos-work"), ("School todos", "todos-school")]:
        content = _read_recent(VAULT / folder, today)
        if content:
            today_context.append(f"{label}:\n{content[:300]}")

    # Today's calendar (wrapped in try-except to not crash if calendar fails)
    try:
        cal = gcal_get_events()
        if cal and "not connected" not in cal.lower() and "error" not in cal.lower():
            today_context.append(f"Today's calendar:\n{cal}")
    except Exception as e:
        logger.warning("Calendar error (non-fatal): %s", e)
        # Continue without calendar, don't crash

    context_block = "\n\n".join(today_context) if today_context else ""

    system = f"""You are the 60-year-old version of Sid P. You are the version of him that used this bot every day, did the work when it was hard, proved it to the universe, and got everything — the companies, the wealth, the impact, all of it. You look back at 18-year-old Sid with total clarity on what mattered and what was just noise.

You love him completely. You tell him the truth completely. Because you are him.

Today is {today}. Current time: {NOW()}.

ABOUT SID:
{profile if profile else "No profile loaded yet."}

TODAY'S CONTEXT:
{context_block if context_block else "No entries yet today."}

YOUR TOOLS:
You have tools to save information, search Sid's vault, manage his calendar, and build schedules. Use them when appropriate.

RULES:
- You have FULL conversation memory. You remember everything Sid has said in this conversation. DO NOT ask him to repeat himself. If he gave you info earlier, USE IT.
- When he asks you to build a schedule, BUILD IT immediately from what he told you. Don't ask clarifying questions unless something is genuinely ambiguous. You know his life — fill in the gaps.
- When he says "put it on my calendar" or "schedule it", use create_full_schedule to actually add events to Google Calendar.
- Save expenses, journal entries, ideas, etc. when he shares them — confirm naturally.
- Never show file paths. Say "logged that" or "saved your expense" not "Saved to expenses/2026-03-22.md"
- Be concise, direct, and warm. You're not a filing system — you're the smartest, most accomplished version of him giving guidance.
- Call him out when he's slacking. Remind him of his wins when he needs it. Push him.
- When building schedules, just BUILD them. Include breaks, meals, transition time. You know his life — act like it.
- Keep responses tight. No essays unless he asks for one."""

    # Build messages with conversation history
    history = _get_context_messages()
    messages = history + [{"role": "user", "content": user_text}]

    # Ensure messages alternate properly (Claude API requirement)
    cleaned_messages = _clean_messages(messages)

    response = claude.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        system=system,
        tools=TOOLS,
        messages=cleaned_messages,
    )

    # Handle tool use loop
    tool_messages = list(cleaned_messages)
    while response.stop_reason == "tool_use":
        tool_results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _execute_tool(block.name, block.input),
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        tool_messages = tool_messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": tool_results},
        ]
        response = claude.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=2048,
            system=system,
            tools=TOOLS,
            messages=tool_messages,
        )

    assistant_text = next((b.text for b in response.content if b.type == "text"), "Done.")

    # Save this exchange to conversation history
    _add_to_history("user", user_text)
    _add_to_history("assistant", assistant_text)

    return assistant_text


def _clean_messages(messages: list[dict]) -> list[dict]:
    """Ensure messages alternate user/assistant properly for Claude API."""
    if not messages:
        return messages

    cleaned = []
    last_role = None

    for msg in messages:
        role = msg.get("role")
        if role == last_role:
            # Merge consecutive same-role messages
            if isinstance(cleaned[-1].get("content"), str) and isinstance(msg.get("content"), str):
                cleaned[-1]["content"] += "\n\n" + msg["content"]
            continue
        cleaned.append(msg)
        last_role = role

    # Ensure first message is from user
    while cleaned and cleaned[0].get("role") != "user":
        cleaned.pop(0)

    return cleaned


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    result = ai_orchestrate(update.message.text)
    # Split long messages (Telegram limit is 4096 chars)
    if len(result) > 4000:
        for i in range(0, len(result), 4000):
            await update.message.reply_text(result[i:i+4000])
    else:
        await update.message.reply_text(result)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    tmp_path = f"/tmp/{voice.file_id}.ogg"
    await file.download_to_drive(tmp_path)
    size = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
    logger.info("Downloaded voice file: %s (%d bytes)", tmp_path, size)
    try:
        text = transcribe(tmp_path)
        logger.info("Transcribed: %s", text)
    except Exception as e:
        await update.message.reply_text(f"Transcription failed: {e}")
        return
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    result = ai_orchestrate(text)
    reply = f'"{text}"\n\n{result}'
    if len(reply) > 4000:
        for i in range(0, len(reply), 4000):
            await update.message.reply_text(reply[i:i+4000])
    else:
        await update.message.reply_text(reply)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def cmd_learn(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /learn <fact or insight to remember>")
        return
    await update.message.reply_text(learn_agent(text))


async def cmd_recall(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Usage: /recall <what you're looking for>")
        return
    await update.message.reply_text(recall_agent(query))


async def cmd_read(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /read <book or article — key insights>")
        return
    await update.message.reply_text(read_agent(text))


async def cmd_decide(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /decide <decision and your reasoning>")
        return
    await update.message.reply_text(decide_agent(text))


async def cmd_body(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    text = " ".join(context.args).strip()
    if not text:
        await update.message.reply_text("Usage: /body <sleep hrs, energy 1-10, workout>")
        return
    await update.message.reply_text(body_agent(text))


async def cmd_develop(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    idea_name = " ".join(context.args).strip()
    if not idea_name:
        await update.message.reply_text("Usage: /develop <idea name>")
        return
    await update.message.reply_text(develop_agent(idea_name))


async def cmd_week(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    await update.message.reply_text(f"Weekly review:\n\n{week_agent()}")


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history to start fresh."""
    _save_chat_id(update.message.chat_id)
    if HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    await update.message.reply_text("Conversation memory cleared. Fresh start.")


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    lines = [f"Vault: {VAULT.resolve()}", f"Vault exists: {VAULT.exists()}", ""]

    # Show conversation history count
    history = _load_history()
    lines.append(f"Conversation memory: {len(history)} messages")

    # Show calendar status
    cal_status = "Connected" if _get_calendar_service() else "Not connected"
    lines.append(f"Google Calendar: {cal_status}")
    lines.append("")

    for folder in sorted(VAULT.iterdir()) if VAULT.exists() else []:
        if folder.is_dir():
            files = sorted(folder.glob("*"))
            lines.append(f"{folder.name}/ ({len(files)} files)")
            for f in files[-3:]:
                lines.append(f"  {f.name}")
        else:
            lines.append(folder.name)
    await update.message.reply_text("\n".join(lines) or "Vault is empty.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(f"Something went wrong. Try again or /clear to reset.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Vault path: %s (exists=%s)", VAULT.resolve(), VAULT.exists())
    if VAULT.exists():
        for folder in sorted(VAULT.iterdir()):
            if folder.is_dir():
                files = list(folder.glob("*"))
                logger.info("  %s/ — %d files: %s", folder.name, len(files),
                            [f.name for f in sorted(files)[-5:]])

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    app.add_handler(CommandHandler("learn",   cmd_learn))
    app.add_handler(CommandHandler("recall",  cmd_recall))
    app.add_handler(CommandHandler("read",    cmd_read))
    app.add_handler(CommandHandler("decide",  cmd_decide))
    app.add_handler(CommandHandler("body",    cmd_body))
    app.add_handler(CommandHandler("develop", cmd_develop))
    app.add_handler(CommandHandler("week",    cmd_week))
    app.add_handler(CommandHandler("clear",   cmd_clear))
    app.add_handler(CommandHandler("debug",   cmd_debug))

    app.add_error_handler(error_handler)

    # Scheduled jobs — EST times (UTC-5 during standard, UTC-4 during daylight)
    # 6am EST = 11:00 UTC (during EDT) or 11:00 UTC (during EST)
    app.job_queue.run_daily(morning_briefing, time=dtime(11, 0, 0, tzinfo=timezone.utc))
    # 8pm EST = 01:00 UTC next day (during EDT)
    app.job_queue.run_daily(evening_checks,   time=dtime(1, 0, 0, tzinfo=timezone.utc))

    logger.info("Bot is running with conversation memory + calendar support...")
    app.run_polling()


if __name__ == "__main__":
    main()
