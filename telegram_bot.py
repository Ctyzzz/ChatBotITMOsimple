import os
import asyncio
import textwrap
from typing import Optional

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)

from itmo_core import (
    ITMOIndex, AI_URL, AI_PRODUCT_URL,
    make_answer_from_chunks, is_offtopic,
    recommend_program, suggest_electives
)

load_dotenv()

TOKEN = os.environ.get("TELEGRAM_TOKEN")

INDEX = ITMOIndex()

HELP = textwrap.dedent("""\
    Привет! Я помогу с двумя с выбором между двумя программами магистратуры ИТМО:
    • AI (Искусственный интеллект)
    • AI Product (Управление ИИ-продуктами)
    
    Команды:
    /load — загрузить страницы и построить индекс
    /ask <вопрос> — ответить по содержанию программ (только релевантные вопросы)
    /choose <профиль> — рекомендация: AI vs AI Product
    /electives <интересы> — подсказать элективы/модули
    /status — статус индекса
    /help — подсказка
""")

def require_args(args: str) -> Optional[str]:
    args = (args or "").strip()
    return args if args else None

async def _typing(ctx: ContextTypes.DEFAULT_TYPE, chat_id: int):
    await ctx.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ready = INDEX.vectorizer is not None
    await update.message.reply_text(
        f"Статус: {'готов' if ready else 'нужна загрузка'} • фрагментов: {len(INDEX.chunks)}"
    )

async def load(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await _typing(context, update.effective_chat.id)
    await update.message.reply_text("Загружаю страницы и строю индекс… Это может занять 10–40 секунд.")
    # Run blocking build in thread
    def _build():
        INDEX.build([AI_URL, AI_PRODUCT_URL])
        return len(INDEX.chunks)
    count = await asyncio.to_thread(_build)
    await update.message.reply_text(f"Готово. Проиндексировано фрагментов: {count}")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if INDEX.vectorizer is None:
        await update.message.reply_text("Сначала выполните /load")
        return
    q = require_args(" ".join(context.args))
    if not q:
        await update.message.reply_text("Использование: /ask <ваш вопрос>")
        return
    if is_offtopic(q):
        await update.message.reply_text(
            "Я отвечаю только на вопросы по двум программам ИТМО (AI и AI Product) и их учебным планам."
        )
        return
    await _typing(context, update.effective_chat.id)
    def _search():
        hits = INDEX.search(q, topk=6)
        ans = make_answer_from_chunks(q, hits) or "Не нашлось релевантного ответа в загруженных материалах."
        return ans
    ans = await asyncio.to_thread(_search)
    # Telegram messages must be <= 4096 chars
    for chunk in textwrap.wrap(ans, width=3800, replace_whitespace=False):
        await update.message.reply_text(chunk)

async def choose(update: Update, context: ContextTypes.DEFAULT_TYPE):
    profile = require_args(" ".join(context.args))
    if not profile:
        await update.message.reply_text("Использование: /choose <кратко опишите опыт и цели>")
        return
    rec, scores = recommend_program(profile)
    await update.message.reply_text(f"{rec}\nОценки: AI={scores['ai_score']}, Product={scores['product_score']}")

async def electives(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if INDEX.vectorizer is None:
        await update.message.reply_text("Сначала выполните /load")
        return
    interests = require_args(" ".join(context.args))
    if not interests:
        await update.message.reply_text("Использование: /electives <интересы>")
        return
    await _typing(context, update.effective_chat.id)
    tips = await asyncio.to_thread(suggest_electives, INDEX, interests)
    if not tips:
        await update.message.reply_text("Не удалось найти элективы по заданным интересам. Попробуйте переформулировать.")
        return
    text = "Подходящие элективы/модули:\n" + "\n".join(f"• {t}" for t in tips)
    for chunk in textwrap.wrap(text, width=3800, replace_whitespace=False):
        await update.message.reply_text(chunk, parse_mode=ParseMode.HTML)

def main():
    if not TOKEN:
        print("Set TELEGRAM_TOKEN env var or put it into .env as TELEGRAM_TOKEN=...")
        raise SystemExit(1)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("load", load))
    app.add_handler(CommandHandler("ask", ask))
    app.add_handler(CommandHandler("choose", choose))
    app.add_handler(CommandHandler("electives", electives))
    print("Bot is running. Press Ctrl+C to stop.")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
