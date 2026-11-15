
import os
import logging
import asyncio
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from google import genai
from google.genai import types


# ---------------- CONFIGURATION ----------------

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN environment variable")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY)

START_TIME = time.monotonic()

# Separate conversation history for each Telegram user.
user_memory = defaultdict(list)
user_locks = defaultdict(asyncio.Lock)

# Usage protection.
user_minute_requests = defaultdict(deque)
user_daily_requests = defaultdict(int)
global_daily_requests = 0
usage_date = datetime.now(timezone.utc).date()

MAX_REQUESTS_PER_MINUTE = 8
MAX_REQUESTS_PER_USER_DAY = 100
MAX_REQUESTS_GLOBAL_DAY = 300

MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 12

SYSTEM_INSTRUCTION = """
You are a helpful, friendly AI assistant inside Telegram.

Automatically identify the language used by the user.
Support English, Khmer, and Chinese.
Reply in the same language as the user's latest message,
unless the user requests another language.

Be clear, useful, polite, and accurate.
If you do not know something, say so.
Do not invent facts.
Keep answers reasonably concise for Telegram.
"""


# ---------------- USAGE LIMITS ----------------

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
        return "You are sending messages too quickly. Please wait a minute."

    if user_daily_requests[user_id] >= MAX_REQUESTS_PER_USER_DAY:
        return "Your daily AI usage limit has been reached. Please try again tomorrow."

    if global_daily_requests >= MAX_REQUESTS_GLOBAL_DAY:
        return "The bot has reached its daily usage limit. Please try again tomorrow."

    requests.append(now)
    user_daily_requests[user_id] += 1
    global_daily_requests += 1

    return None


# ---------------- TELEGRAM COMMANDS ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hello! 👋\n\n"
        "I'm your AI assistant powered by Google Gemini.\n\n"
        "I can communicate in English, Khmer, and Chinese.\n\n"
        "Just send me a message to start chatting.\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show help\n"
        "/reset - Clear your conversation memory\n"
        "/ping - Check bot status"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 AI Assistant Help\n\n"
        "• Send any text message to chat with AI.\n"
        "• English, Khmer, and Chinese are supported.\n"
        "• Each Telegram user has separate conversation memory.\n\n"
        "Commands:\n"
        "/start - Start\n"
        "/help - Help\n"
        "/reset - Clear your conversation memory\n"
        "/ping - Check bot status"
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id

    async with user_locks[user_id]:
        user_memory[user_id] = []

    await update.message.reply_text(
        "Your conversation memory has been cleared. 🧹\n"
        "You can start a new conversation now."
    )


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uptime = int(time.monotonic() - START_TIME)

    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)

    await update.message.reply_text(
        "🟢 Bot is online!\n\n"
        f"Uptime: {hours}h {minutes}m {seconds}s\n"
        f"AI model: {MODEL}"
    )


# ---------------- AI CHAT ----------------

async def split_and_send(message, text):
    # Telegram message limit is 4096 characters.
    for start_index in range(0, len(text), 4000):
        await message.reply_text(text[start_index:start_index + 4000])


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

    # Keep one request at a time per user.
    async with user_locks[user_id]:
        limit_message = check_usage(user_id)

        if limit_message:
            await message.reply_text(limit_message)
            return

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )

        history = user_memory[user_id]

        # Build request without changing saved history until success.
        contents = list(history)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        )

        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    max_output_tokens=1200,
                    temperature=0.7,
                ),
            )

            answer = response.text

            if not answer:
                await message.reply_text(
                    "I couldn't generate a response. Please try again."
                )
                return

            # Save conversation only after a successful response.
            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            )
            history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=answer)],
                )
            )

            # Keep only the latest 12 messages.
            user_memory[user_id] = history[-MAX_HISTORY_MESSAGES:]

            await split_and_send(message, answer)

        except Exception as error:
            logger.error(
                "Gemini request failed (%s)",
                type(error).__name__,
            )

            await message.reply_text(
                "Sorry, the AI service is temporarily unavailable. "
                "Please try again later."
            )


# ---------------- ERROR HANDLER ----------------

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    logger.error(
        "Telegram handler error (%s)",
        type(context.error).__name__ if context.error else "Unknown",
    )


# ---------------- START BOT ----------------

def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(CommandHandler("ping", ping))

    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, chat)
    )

    application.add_error_handler(error_handler)

    logger.info("Starting Telegram Gemini AI bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
