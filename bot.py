import os
import time
import asyncio
import logging
from datetime import datetime, timezone
from collections import defaultdict, deque

from google import genai
from google.genai import types

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)


# ==================================================
# CONFIGURATION
# ==================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

START_TIME = time.monotonic()

SYSTEM_INSTRUCTION = """
You are a helpful and friendly AI assistant on Telegram.

Automatically detect the language of the user's latest message.

Support:
- English
- Khmer
- Chinese (Simplified and Traditional)

Reply in the same language as the user's latest message,
unless they request another language.

Give accurate, useful and clear answers.
Never invent facts.
Keep answers concise unless more detail is requested.
Use simple formatting suitable for Telegram.
"""

# ==================================================
# PERFORMANCE SETTINGS
# ==================================================

MAX_HISTORY_MESSAGES = 6
MAX_MESSAGE_LENGTH = 4000
MAX_OUTPUT_TOKENS = 800

API_TIMEOUT = 45

# Limit simultaneous requests to protect the hosting resources.
API_SEMAPHORE = asyncio.Semaphore(5)

# ==================================================
# USAGE PROTECTION
# ==================================================

MAX_REQUESTS_PER_MINUTE = 8
MAX_REQUESTS_PER_USER_DAY = 100
MAX_REQUESTS_GLOBAL_DAY = 300

user_minute_requests = defaultdict(deque)
user_daily_requests = defaultdict(int)

global_daily_requests = 0
usage_date = datetime.now(timezone.utc).date()

# ==================================================
# USER CONVERSATION MEMORY
# ==================================================

user_memory = defaultdict(list)
user_locks = defaultdict(asyncio.Lock)


def refresh_daily_usage():
    global usage_date, global_daily_requests

    today = datetime.now(timezone.utc).date()

    if today != usage_date:
        usage_date = today
        global_daily_requests = 0
        user_daily_requests.clear()


def check_usage(user_id):
    global global_daily_requests

    refresh_daily_usage()

    now = time.monotonic()
    requests = user_minute_requests[user_id]

    while requests and now - requests[0] >= 60:
        requests.popleft()

    if len(requests) >= MAX_REQUESTS_PER_MINUTE:
        return "Too many messages. Please wait one minute."

    if user_daily_requests[user_id] >= MAX_REQUESTS_PER_USER_DAY:
        return "You have reached your daily limit. Please try again tomorrow."

    if global_daily_requests >= MAX_REQUESTS_GLOBAL_DAY:
        return "The bot has reached its daily usage limit. Please try again tomorrow."

    requests.append(now)
    user_daily_requests[user_id] += 1
    global_daily_requests += 1

    return None


# ==================================================
# TELEGRAM COMMANDS
# ==================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! 👋 Welcome to my AI assistant.\n\n"
        "Powered by Google Gemini.\n"
        "Languages: English, Khmer and Chinese.\n\n"
        "Send me a message to start chatting.\n\n"
        "Commands:\n"
        "/start - Start bot\n"
        "/help - Help\n"
        "/reset - Clear your memory\n"
        "/ping - Check bot status"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Assistant\n\n"
        "• Chat naturally with AI.\n"
        "• Automatic language detection.\n"
        "• Private conversation memory for each user.\n\n"
        "Commands:\n"
        "/start - Start\n"
        "/help - Help\n"
        "/reset - Clear memory\n"
        "/ping - Check status"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    async with user_locks[user_id]:
        user_memory[user_id] = []

    await update.message.reply_text(
        "🧹 Your conversation memory has been cleared!"
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = int(time.monotonic() - START_TIME)

    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)

    await update.message.reply_text(
        "🟢 Bot is online!\n\n"
        f"Uptime: {hours}h {minutes}m {seconds}s\n"
        f"Model: {MODEL}"
    )


# ==================================================
# GEMINI STREAMING
# ==================================================

async def generate_stream(contents, placeholder):
    """
    Generate Gemini response asynchronously.
    Update the Telegram message periodically.
    """

    full_answer = ""
    last_edit = time.monotonic()
    displayed_text = ""

    stream = await client.aio.models.generate_content_stream(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=0.7,
        ),
    )

    async for chunk in stream:

        if chunk.text:
            full_answer += chunk.text

        # Update the Telegram message periodically.
        now = time.monotonic()

        if (
            full_answer
            and now - last_edit >= 1.2
            and len(full_answer) >= 40
        ):
            preview = full_answer[:3900]

            if preview != displayed_text:
                try:
                    await placeholder.edit_text(preview)
                    displayed_text = preview
                except Exception as error:
                    logger.warning(
                        "Telegram progress update failed: %s",
                        type(error).__name__,
                    )

            last_edit = now

    return full_answer


async def send_final_answer(placeholder, answer):
    """
    Edit the initial message with the final answer.
    Send additional Telegram messages if necessary.
    """

    if len(answer) <= 3900:
        await placeholder.edit_text(answer)
        return

    first_part = answer[:3900]
    await placeholder.edit_text(first_part)

    remaining = answer[3900:]

    for index in range(0, len(remaining), 4000):
        await placeholder.get_bot().send_message(
            chat_id=placeholder.chat_id,
            text=remaining[index:index + 4000],
            reply_to_message_id=placeholder.message_id,
        )


# ==================================================
# MAIN AI CHAT HANDLER
# ==================================================

async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):

    message = update.message
    user_id = update.effective_user.id

    prompt = (message.text or "").strip()

    if not prompt:
        return

    if len(prompt) > MAX_MESSAGE_LENGTH:
        await message.reply_text(
            f"Your message is too long. Maximum: {MAX_MESSAGE_LENGTH} characters."
        )
        return

    async with user_locks[user_id]:

        limit_message = check_usage(user_id)

        if limit_message:
            await message.reply_text(limit_message)
            return

        # Show immediate feedback.
        placeholder = await message.reply_text("🤔 Thinking...")

        try:
            await context.bot.send_chat_action(
                chat_id=update.effective_chat.id,
                action=ChatAction.TYPING,
            )

            history = user_memory[user_id]

            contents = list(history)

            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt)
                    ],
                )
            )

            async with API_SEMAPHORE:

                answer = await asyncio.wait_for(
                    generate_stream(contents, placeholder),
                    timeout=API_TIMEOUT,
                )

            if not answer.strip():
                await placeholder.edit_text(
                    "Sorry, I couldn't generate an answer. Please try again."
                )
                return

            # Save history only after a successful response.
            history.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt)
                    ],
                )
            )

            history.append(
                types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text=answer)
                    ],
                )
            )

            user_memory[user_id] = history[-MAX_HISTORY_MESSAGES:]

            await send_final_answer(placeholder, answer)

        except asyncio.TimeoutError:

            logger.warning("Gemini request timed out.")

            await placeholder.edit_text(
                "⏳ The AI is taking too long. Please try again."
            )

        except Exception as error:

            logger.error(
                "AI request failed: %s",
                type(error).__name__,
            )

            try:
                await placeholder.edit_text(
                    "⚠️ The AI service is temporarily unavailable. "
                    "Please try again later."
                )
            except Exception:
                pass


# ==================================================
# ERROR HANDLER
# ==================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):

    error = context.error

    logger.error(
        "Telegram error: %s",
        type(error).__name__ if error else "Unknown",
    )


# ==================================================
# START APPLICATION
# ==================================================

def main():

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start)
    )

    application.add_handler(
        CommandHandler("help", help_command)
    )

    application.add_handler(
        CommandHandler("reset", reset)
    )

    application.add_handler(
        CommandHandler("ping", ping)
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            chat,
        )
    )

    application.add_error_handler(error_handler)

    logger.info("Gemini Telegram bot starting...")
    logger.info("Model: %s", MODEL)

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
