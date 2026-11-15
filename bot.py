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


# =========================================================
# 1. CONFIGURATION
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")


# =========================================================
# 2. LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("telegram_gemini_bot")

# Prevent HTTP logs from exposing Telegram bot tokens.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)


# =========================================================
# 3. GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

START_TIME = time.monotonic()


# =========================================================
# 4. AI SETTINGS
# =========================================================

SYSTEM_INSTRUCTION = """
You are a helpful, friendly and intelligent AI assistant
inside Telegram.

Automatically detect the language of each user's message.

Supported languages:
- English
- Khmer
- Chinese (Simplified and Traditional)

Always reply in the same language as the user's latest
message unless they explicitly request another language.

Your responses should be:
- Accurate
- Natural
- Friendly
- Easy to understand
- Suitable for Telegram

Keep answers concise unless the user asks for details.

Never invent facts. If you are uncertain, explain that.
"""

MAX_MESSAGE_LENGTH = 4000
MAX_OUTPUT_TOKENS = 800

MAX_HISTORY_MESSAGES = 6

MAX_API_ATTEMPTS = 3
TOTAL_REQUEST_TIMEOUT = 90

# Avoid overwhelming a small hosting service.
API_SEMAPHORE = asyncio.Semaphore(3)


# =========================================================
# 5. USAGE PROTECTION
# =========================================================

MAX_REQUESTS_PER_MINUTE = 8
MAX_REQUESTS_PER_USER_DAY = 100
MAX_REQUESTS_GLOBAL_DAY = 300

user_minute_requests = defaultdict(deque)
user_daily_requests = defaultdict(int)

global_daily_requests = 0

usage_date = datetime.now(timezone.utc).date()


def refresh_daily_usage():

    global usage_date
    global global_daily_requests

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

        return (
            "⏳ You are sending messages too quickly. "
            "Please wait one minute."
        )

    if user_daily_requests[user_id] >= MAX_REQUESTS_PER_USER_DAY:

        return (
            "You have reached your daily usage limit. "
            "Please try again tomorrow."
        )

    if global_daily_requests >= MAX_REQUESTS_GLOBAL_DAY:

        return (
            "The bot has reached its daily usage limit. "
            "Please try again tomorrow."
        )

    requests.append(now)

    user_daily_requests[user_id] += 1
    global_daily_requests += 1

    return None


# =========================================================
# 6. CONVERSATION MEMORY
# =========================================================

user_memory = defaultdict(list)

# Prevent simultaneous requests from modifying one user's
# conversation history at the same time.
user_locks = defaultdict(asyncio.Lock)


# =========================================================
# 7. TELEGRAM COMMANDS
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "Hello! 👋 Welcome to my AI assistant.\n\n"
        "Powered by Google Gemini.\n\n"
        "Languages:\n"
        "🇬🇧 English\n"
        "🇰🇭 Khmer\n"
        "🇨🇳 Chinese\n\n"
        "Send me a message to start chatting.\n\n"
        "Commands:\n"
        "/start - Start bot\n"
        "/help - Help\n"
        "/reset - Clear conversation memory\n"
        "/ping - Check bot status"
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    await update.message.reply_text(
        "🤖 AI Assistant Help\n\n"
        "• Chat naturally with AI.\n"
        "• Automatic language detection.\n"
        "• English, Khmer and Chinese.\n"
        "• Separate conversation memory.\n\n"
        "Commands:\n"
        "/start - Start\n"
        "/help - Help\n"
        "/reset - Clear memory\n"
        "/ping - Check status"
    )


async def reset(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    user_id = update.effective_user.id

    async with user_locks[user_id]:

        user_memory[user_id] = []

    await update.message.reply_text(
        "🧹 Your conversation memory has been cleared!"
    )


async def ping(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    uptime = int(time.monotonic() - START_TIME)

    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)

    await update.message.reply_text(
        "🟢 Bot is online!\n\n"
        f"Uptime: {hours}h {minutes}m {seconds}s\n"
        f"AI Model: {MODEL}"
    )


# =========================================================
# 8. ERROR CLASSIFICATION
# =========================================================

def get_error_status(error):

    # Different SDK versions may expose status differently.
    for attribute in ("code", "status_code"):

        value = getattr(error, attribute, None)

        if value is not None:

            try:
                return int(value)
            except (TypeError, ValueError):
                pass

    response = getattr(error, "response", None)

    if response is not None:

        value = getattr(response, "status_code", None)

        if value is not None:
            return value

    return None


def is_temporary_error(error):

    status = get_error_status(error)

    if status in (408, 429, 500, 502, 503, 504):
        return True

    # Fallback for SDK exceptions without exposed status codes.
    name = type(error).__name__.lower()

    temporary_names = (
        "servererror",
        "toomanyrequests",
        "serviceunavailable",
        "deadlineexceeded",
        "timeouterror",
    )

    return any(item in name for item in temporary_names)


# =========================================================
# 9. STREAMING GENERATION
# =========================================================

async def generate_once(contents, placeholder):

    full_answer = ""

    last_edit = time.monotonic()
    last_displayed = ""

    # Important: the async SDK returns an async iterator
    # after awaiting generate_content_stream.
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

        now = time.monotonic()

        # Update Telegram while Gemini is generating.
        if (
            full_answer
            and len(full_answer) >= 40
            and now - last_edit >= 1.5
        ):

            preview = full_answer[:3900]

            if preview != last_displayed:

                try:

                    await placeholder.edit_text(preview)

                    last_displayed = preview

                except Exception as edit_error:

                    logger.warning(
                        "Progress update failed: %s",
                        type(edit_error).__name__,
                    )

            last_edit = now

    return full_answer


# =========================================================
# 10. AUTOMATIC RETRY
# =========================================================

async def generate_with_retry(contents, placeholder):

    for attempt in range(MAX_API_ATTEMPTS):

        try:

            logger.info(
                "Gemini request attempt %s/%s",
                attempt + 1,
                MAX_API_ATTEMPTS,
            )

            answer = await generate_once(
                contents,
                placeholder,
            )

            return answer

        except Exception as error:

            # Do not retry after receiving part of an answer:
            # it could create duplicated or confusing responses.
            if not is_temporary_error(error):

                logger.error(
                    "Non-retryable Gemini error: %s",
                    type(error).__name__,
                )

                raise

            if attempt == MAX_API_ATTEMPTS - 1:

                logger.error(
                    "Gemini retries exhausted: %s",
                    type(error).__name__,
                )

                raise

            delay = 2 ** (attempt + 1)

            logger.warning(
                "Temporary Gemini error (%s). "
                "Retrying in %s seconds.",
                type(error).__name__,
                delay,
            )

            await placeholder.edit_text(
                "⏳ Gemini is temporarily busy.\n"
                f"Retrying ({attempt + 2}/{MAX_API_ATTEMPTS})..."
            )

            await asyncio.sleep(delay)

    return ""


# =========================================================
# 11. SEND FINAL ANSWER
# =========================================================

async def send_final_answer(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    placeholder,
    answer: str
):

    # Telegram messages have a maximum length.
    chunks = [
        answer[index:index + 3900]
        for index in range(0, len(answer), 3900)
    ]

    if not chunks:
        chunks = ["I couldn't generate a response."]

    try:

        await placeholder.edit_text(chunks[0])

    except Exception:

        # If the first response was already displayed,
        # avoid failing the whole conversation.
        pass

    # Send any additional parts.
    for part in chunks[1:]:

        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=part,
        )


# =========================================================
# 12. MAIN AI CHAT HANDLER
# =========================================================

async def chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    message = update.message

    user_id = update.effective_user.id

    prompt = (message.text or "").strip()

    if not prompt:
        return

    if len(prompt) > MAX_MESSAGE_LENGTH:

        await message.reply_text(
            f"Your message is too long. "
            f"Maximum: {MAX_MESSAGE_LENGTH} characters."
        )

        return

    async with user_locks[user_id]:

        limit_message = check_usage(user_id)

        if limit_message:

            await message.reply_text(limit_message)

            return

        # Immediate response so users know the bot is working.
        placeholder = await message.reply_text(
            "🤔 Thinking..."
        )

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
                    generate_with_retry(
                        contents,
                        placeholder,
                    ),
                    timeout=TOTAL_REQUEST_TIMEOUT,
                )

            if not answer or not answer.strip():

                await placeholder.edit_text(
                    "Sorry, I couldn't generate an answer. "
                    "Please try again."
                )

                return

            # Save memory only after successful generation.
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

            user_memory[user_id] = history[
                -MAX_HISTORY_MESSAGES:
            ]

            await send_final_answer(
                update,
                context,
                placeholder,
                answer,
            )

        except asyncio.TimeoutError:

            logger.warning("Gemini request timed out.")

            await placeholder.edit_text(
                "⏳ The AI is taking too long. "
                "Please try again."
            )

        except Exception as error:

            logger.error(
                "AI request failed: %s",
                type(error).__name__,
            )

            try:

                await placeholder.edit_text(
                    "⚠️ The AI service is temporarily unavailable.\n\n"
                    "Please try again in a little while."
                )

            except Exception:
                pass


# =========================================================
# 13. GLOBAL ERROR HANDLER
# =========================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE
):

    error = context.error

    logger.error(
        "Telegram application error: %s",
        type(error).__name__ if error else "Unknown",
    )


# =========================================================
# 14. START APPLICATION
# =========================================================

def main():

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    # Commands
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

    # Normal text messages
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            chat,
        )
    )

    application.add_error_handler(
        error_handler
    )

    logger.info("Telegram Gemini AI bot starting...")
    logger.info("Model configured: %s", MODEL)

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":

    main()
