!pip install python-telegram-bot
import logging
import nest_asyncio
import asyncio
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
import torch
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, CallbackContext, CallbackQueryHandler

# Применение nest_asyncio для работы в Jupyter или Google Colab
nest_asyncio.apply()

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация токенизаторов и моделей
classification_model_name = "bert-base-uncased"
generation_model_name = "gpt2"

classification_tokenizer = BertTokenizer.from_pretrained(classification_model_name)
classification_model = BertForSequenceClassification.from_pretrained(classification_model_name)

generation_tokenizer = GPT2Tokenizer.from_pretrained(generation_model_name)
generation_model = GPT2LMHeadModel.from_pretrained(generation_model_name)

current_model = "classification"  # Хранение текущей модели

# Классы для классификации
classes = {0: "Работа", 1: "Учёба", 2: "Спорт", 3: "Музыка"}

async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Привет! Напиши мне сообщение, и я скажу, к какой теме оно относится.")

async def model(update: Update, context: CallbackContext) -> None:
    keyboard = [
        [InlineKeyboardButton("Классификация (BERT)", callback_data='classification')],
        [InlineKeyboardButton("Генерация (GPT-2)", callback_data='generation')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)

async def button(update: Update, context: CallbackContext) -> None:
    global current_model
    query = update.callback_query
    await query.answer()
    
    if query.data in ["classification", "generation"]:
        current_model = query.data
        await query.edit_message_text(text=f"Выбрана модель: {current_model}")

async def checkmodel(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text(f"Текущая модель: {current_model}")

async def generate_command(update: Update, context: CallbackContext) -> None:
    if not context.args:
        await update.message.reply_text("Пожалуйста, введите текст после команды /generate.")
        return

    user_input = " ".join(context.args)  # Объединяем аргументы в один текст
    logging.info(f"Получен запрос на генерацию текста: {user_input}")

    inputs = generation_tokenizer.encode(user_input, return_tensors="pt")

    # Генерация текста
    try:
        output_sequences = generation_model.generate(inputs, max_length=50, num_return_sequences=1)
        generated_text = generation_tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        await update.message.reply_text(f"Сгенерированный текст: {generated_text}")
    except Exception as e:
        logging.error(f"Ошибка генерации текста: {e}")
        await update.message.reply_text("Произошла ошибка при генерации текста.")

async def classify_text(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text
    inputs = classification_tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)

    classification_model.eval()
    with torch.no_grad():
        outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()

    topic = classes.get(predicted_class, "Неизвестная тема")
    await update.message.reply_text(f"Ваше сообщение относится к теме: {topic}")

async def handle_message(update: Update, context: CallbackContext) -> None:
    if current_model == "generation":
        await generate_command(update, context)
    else:
        await classify_text(update, context)

async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "/start - старт бота\n"
        "/model - выбор модели\n"
        "/checkmodel - посмотреть текущую модель\n"
        "/generate <текст> - сгенерировать текст по контексту\n"
        "/help - вывести список доступных команд"
    )
    await update.message.reply_text(help_text)

def main() -> None:
    application = ApplicationBuilder().token("private").build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("model", model))
    application.add_handler(CommandHandler("checkmodel", checkmodel))
    application.add_handler(CommandHandler("generate", generate_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()

if __name__ == '__main__':
    main()
