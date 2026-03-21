import os
import json
import logging
import uuid
from datetime import date, timedelta, time as dtime, timezone
from pathlib import Path

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
)
import openai
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WHISPER_KEY    = os.environ["WHISPER_KEY"]
ANTHROPIC_KEY  = os.environ["ANTHROPIC_KEY"]

openai.api_key = WHISPER_KEY
claude = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

VAULT        = Path(__file__).parent / "vault"
CHAT_ID_FILE = VAULT / "chat_id.txt"
TODAY        = lambda: date.today().isoformat()


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
        logger.debug("_read_days: folder does not exist: %s", folder)
        return ""
    cutoff = date.today() - timedelta(days=days)
    parts = []
    for f in sorted(folder.glob("*.md"), reverse=True):
        try:
            file_date = date.fromisoformat(f.stem)
        except ValueError:
            logger.debug("_read_days: skipping non-date file %s", f.name)
            continue
        logger.debug("_read_days: %s — date=%s cutoff=%s included=%s", f.name, file_date, cutoff, file_date >= cutoff)
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
    with open(file_path, "rb") as f:
        result = openai.audio.transcriptions.create(model="whisper-1", file=f)
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
    return f"Got it! Saved to {category}/{entry_date}.md"


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
    return f"Got it! Saved to {folder}/{entry_date}.md"


def scheduler_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "habits" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Got it! Saved to habits/{entry_date}.md"


def ideas_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "ideas" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Got it! Saved to ideas/{entry_date}.md"


def journal_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "journal" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Got it! Saved to journal/{entry_date}.md"


# ---------------------------------------------------------------------------
# Command agents (LLM-powered, only when explicitly invoked)
# ---------------------------------------------------------------------------

def focus_agent() -> str:
    today = TODAY()
    parts = []
    profile_path = VAULT / "profile.md"
    if profile_path.exists():
        profile = profile_path.read_text(encoding="utf-8").strip()
        if profile:
            parts.append(f"Profile:\n{profile[:600]}")
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
    """Save a fact/insight with spaced repetition schedule."""
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
    return "Got it! Saved to knowledge. First review tomorrow."


def recall_agent(query: str) -> str:
    """Search vault files for a query and synthesize with Claude."""
    folders = [
        "journal", "ideas", "todos-work", "todos-school",
        "habits", "expenses", "income", "reading-log", "decisions", "body",
    ]
    matches = []
    query_lower = query.lower()
    for folder in folders:
        for f in (VAULT / folder).glob("*.md"):
            content = f.read_text(encoding="utf-8")
            if query_lower in content.lower():
                matches.append(f"[{folder}/{f.name}]\n{content[:500]}")

    # Search knowledge index too
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
    """Save a reading note and extract key insight."""
    entry_date = TODAY()
    insight = _call_claude(
        system="Extract the single most important insight from this reading note in one sentence.",
        user=text,
        max_tokens=100,
    )
    target = VAULT / "reading-log" / f"{entry_date}.md"
    _append(target, f"{text}\n\n**Key insight:** {insight}", entry_date)
    return f"Got it! Saved to reading-log.\nKey insight: {insight}"


def decide_agent(text: str) -> str:
    """Save a decision to the decision log."""
    entry_date = TODAY()
    target = VAULT / "decisions" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return "Got it! Decision logged. I'll resurface this in a month for outcome review."


def body_agent(text: str) -> str:
    """Save health/performance data."""
    entry_date = TODAY()
    target = VAULT / "body" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Got it! Body log saved for {entry_date}."


def develop_agent(idea_name: str) -> str:
    """Generate Socratic questions for an idea and track its maturity."""
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
    """Generate a weekly review digest."""
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
        p = profile_path.read_text(encoding="utf-8").strip()[:500]
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
    for f in sorted((VAULT / "journal").glob("*.md"), reverse=True)[:3]:
        journal_snippets.append(f.read_text(encoding="utf-8").strip()[:200])
    if journal_snippets:
        parts.append("Recent journal:\n" + "\n---\n".join(journal_snippets))

    reading_files = sorted((VAULT / "reading-log").glob("*.md"), reverse=True)
    if reading_files:
        parts.append(f"Recent reading:\n{reading_files[0].read_text(encoding='utf-8').strip()[:300]}")

    brief = _call_claude(
        system=(
            "You are a personal assistant giving a morning intelligence brief. "
            "Based on the user's context, provide: "
            "1) Top 3 priorities for today (numbered), "
            "2) One habit to focus on today, "
            "3) One sentence connecting something from their recent reading/learning to something they're working on. "
            "Be direct and motivating. Max 200 words."
        ),
        user="\n\n".join(parts) if parts else "No context yet — encourage them to start logging.",
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
                    "Include: observed strengths, current apparent priorities (based on behavior, not stated goals), "
                    "recurring patterns, and any contradictions between stated goals and actual behavior. "
                    "Be honest and specific. Write in second person. Max 300 words."
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
                text=f"Profile updated from your last 14 days:\n\n{updated}"
            )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FINANCE_KEYWORDS = {"money", "expense", "spent", "earned", "income", "paid", "cost", "bought", "price", "bill", "salary"}
TASK_KEYWORDS    = {"todo", "task", "need to", "school", "work", "homework", "assignment", "meeting", "deadline"}
SCHED_KEYWORDS   = {"schedule", "habit", "routine", "plan", "morning", "evening", "weekly"}
IDEA_KEYWORDS    = {"idea", "future", "someday", "maybe", "concept", "what if", "could", "brainstorm"}
JOURNAL_KEYWORDS = {"today was", "today i ", "i felt", "i feel", "feeling", "my day", "journal", "diary", "reflecting", "reflection", "mood"}

FOCUS_PHRASES  = {"what should i focus", "what to focus", "focus today", "what should i do today",
                   "priorities today", "what's on my plate", "what to work on", "daily focus", "what do i need to do"}
LEARN_PHRASES  = {"i learned", "i just learned", "i want to remember", "want to remember",
                   "note to self", "key insight", "important lesson", "remember that", "remember this"}
READ_PHRASES   = {"i read", "i'm reading", "i finished reading", "just finished reading",
                   "key takeaway", "the book says", "this article", "reading log"}
DECIDE_PHRASES = {"i decided", "i've decided", "made a decision", "my decision is",
                   "i'm choosing to", "going with", "i'm going to go with", "decision:"}
BODY_PHRASES   = {"i slept", "slept for", "hours of sleep", "energy level", "energy is",
                   "worked out", "hit the gym", "skipped the gym", "didn't work out", "no workout", "my workout"}
RECALL_PHRASES  = {"what did i", "remind me", "what have i saved", "search my",
                    "look up", "find my notes", "what do i know about", "have i written about"}
WEEK_PHRASES    = {"weekly review", "how was my week", "week in review", "weekly summary", "review my week"}
QUESTION_WORDS  = {"what ", "how much", "how many", "show me", "tell me", "did i ", "have i ",
                    "how did", "how do", "when did", "where did", "which ", "who ", "why did"}


def _natural_recall(text: str) -> str:
    """Extract the search topic from a natural language recall request, then search."""
    topic = _call_claude(
        system="Extract the search topic from this query in 1-5 words. Output only the search terms, nothing else.",
        user=text,
        max_tokens=20,
    ).strip()
    return recall_agent(topic)


def _is_question(lower: str) -> bool:
    if lower.rstrip().endswith("?"):
        return True
    return any(lower.startswith(qw) or f" {qw}" in lower for qw in QUESTION_WORDS)


def orchestrate(text: str) -> str:
    lower = text.lower()
    if any(kw in lower for kw in FOCUS_PHRASES):
        return focus_agent()
    if any(kw in lower for kw in RECALL_PHRASES):
        return _natural_recall(text)
    if _is_question(lower):
        return _natural_recall(text)
    if any(kw in lower for kw in WEEK_PHRASES):
        return week_agent()
    if any(kw in lower for kw in LEARN_PHRASES):
        return learn_agent(text)
    if any(kw in lower for kw in READ_PHRASES):
        return read_agent(text)
    if any(kw in lower for kw in DECIDE_PHRASES):
        return decide_agent(text)
    if any(kw in lower for kw in BODY_PHRASES):
        return body_agent(text)
    if any(kw in lower for kw in FINANCE_KEYWORDS):
        return finance_agent(text)
    if any(kw in lower for kw in TASK_KEYWORDS):
        return task_agent(text)
    if any(kw in lower for kw in JOURNAL_KEYWORDS):
        return journal_agent(text)
    if any(kw in lower for kw in SCHED_KEYWORDS):
        return scheduler_agent(text)
    if any(kw in lower for kw in IDEA_KEYWORDS):
        return ideas_agent(text)
    return journal_agent(text)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    result = orchestrate(update.message.text)
    await update.message.reply_text(result)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    voice = update.message.voice
    file = await context.bot.get_file(voice.file_id)
    tmp_path = f"/tmp/{voice.file_id}.ogg"
    await file.download_to_drive(tmp_path)
    try:
        text = transcribe(tmp_path)
        logger.info("Transcribed: %s", text)
    except Exception as e:
        await update.message.reply_text(f"Transcription failed: {e}")
        return
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    result = orchestrate(text)
    await update.message.reply_text(f'Transcribed: "{text}"\n\n{result}')


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


async def cmd_debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _save_chat_id(update.message.chat_id)
    lines = [f"Vault: {VAULT.resolve()}", f"Vault exists: {VAULT.exists()}", ""]
    for folder in sorted(VAULT.iterdir()) if VAULT.exists() else []:
        if folder.is_dir():
            files = sorted(folder.glob("*"))
            lines.append(f"{folder.name}/ ({len(files)} files)")
            for f in files[-3:]:  # show last 3
                lines.append(f"  {f.name}")
        else:
            lines.append(folder.name)
    await update.message.reply_text("\n".join(lines) or "Vault is empty.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text(f"Error: {type(context.error).__name__}: {context.error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Startup vault diagnostic
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
    app.add_handler(CommandHandler("debug",   cmd_debug))

    app.add_error_handler(error_handler)

    # Scheduled jobs — all times UTC, adjust hour for your timezone
    app.job_queue.run_daily(morning_briefing, time=dtime(6,  0, 0, tzinfo=timezone.utc))
    app.job_queue.run_daily(evening_checks,   time=dtime(20, 0, 0, tzinfo=timezone.utc))

    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
