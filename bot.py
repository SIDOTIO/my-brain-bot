import os
import logging
from datetime import date
from pathlib import Path

from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
import openai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
WHISPER_KEY = os.environ["WHISPER_KEY"]

openai.api_key = WHISPER_KEY

VAULT = Path(__file__).parent / "vault"
TODAY = lambda: date.today().isoformat()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(file_path: str) -> str:
    with open(file_path, "rb") as f:
        result = openai.audio.transcriptions.create(model="whisper-1", file=f)
    return result.text


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def finance_agent(text: str) -> str:
    entry_date = TODAY()
    # Try to detect income vs expense
    lower = text.lower()
    if any(w in lower for w in ("earned", "income", "received", "salary", "paid me")):
        target = VAULT / "income" / f"{entry_date}.md"
        category = "income"
    else:
        target = VAULT / "expenses" / f"{entry_date}.md"
        category = "expenses"

    _append(target, text, entry_date)
    return f"Saved to {category}/{entry_date}.md"


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
    return f"Saved to {folder}/{entry_date}.md"


def scheduler_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "habits" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Saved to habits/{entry_date}.md"


def ideas_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "ideas" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Saved to ideas/{entry_date}.md"


def journal_agent(text: str) -> str:
    entry_date = TODAY()
    target = VAULT / "journal" / f"{entry_date}.md"
    _append(target, text, entry_date)
    return f"Saved to journal/{entry_date}.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _append(path: Path, text: str, entry_date: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        if path.stat().st_size == 0 if path.exists() else False:
            pass
        f.write(f"\n## {entry_date}\n\n{text}\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

FINANCE_KEYWORDS = {"money", "expense", "spent", "earned", "income", "paid", "cost", "bought", "price", "bill", "salary"}
TASK_KEYWORDS    = {"todo", "task", "need to", "school", "work", "homework", "assignment", "meeting", "deadline"}
SCHED_KEYWORDS   = {"schedule", "today", "tomorrow", "habit", "routine", "plan", "morning", "evening", "weekly"}
IDEA_KEYWORDS    = {"idea", "future", "someday", "maybe", "concept", "what if", "could", "brainstorm"}


def orchestrate(text: str) -> str:
    lower = text.lower()

    if any(kw in lower for kw in FINANCE_KEYWORDS):
        return finance_agent(text)
    if any(kw in lower for kw in TASK_KEYWORDS):
        return task_agent(text)
    if any(kw in lower for kw in SCHED_KEYWORDS):
        return scheduler_agent(text)
    if any(kw in lower for kw in IDEA_KEYWORDS):
        return ideas_agent(text)
    return journal_agent(text)


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    result = orchestrate(text)
    await update.message.reply_text(f"Got it! {result}")


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    await update.message.reply_text(f'Transcribed: "{text}"\n\nSaved! {result}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    logger.info("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
