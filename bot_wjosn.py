import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import CommandHandler, MessageHandler, CallbackQueryHandler, filters, CallbackContext, ApplicationBuilder
import logging
import json

# Настройка логгирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelень)s - %(message)s', level=logging.INFO)

# Загрузка данных из JSON файла
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Добавление нескольких вариантов приветствия от пользователя
user_greetings = ["Здаров", "Ку", "Хай", "Как ты?", "Как дела?", "Как дел?", "Че как ты?", "Привет", "Ну привет", "здравствуй", "добрый день", "добрый вечер", "здравствуйте", "доброе утро", "доброго времени суток", "hello", "hi", "good day", "good evening", "good morning", "greetings", "hey"]

# Добавление вариантов приветствия от пользователя в словарь data
for greeting in user_greetings:
    data[greeting] = "Привет! Я бот-менеджер по консультированию абитурентов университета КазУТБ. Как я могу помочь вам?"

# Добавление нескольких вариантов спасибо от пользователя
user_thanks = ["Спс", "Спасибо", "от души", "Понятно спасибо", "Понятно", "Прекрасно", "Я все понял", "Спасиб", "Спасибки", "Sps", "TY", "thank you", "Spasibo", "Все понял", "Очень информативно", "Спасибо за консультирование", "Thank for help", "Okay", "Ok"]

# Добавление вариантов спасибо от пользователя в словарь data
for thanks in user_thanks:
    data[thanks] = "Не за что! Если у вас еще остались вопросы, не стесняйтесь обращаться."

# Функция для отправки PDF файла с правилами на русском языке
async def send_rule_pdf_rus(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    document_path = 'rule_rus.pdf'  # Путь к файлу с правилами

    try:
        logging.info("Отправка предварительного сообщения")
        await context.bot.send_message(chat_id=chat_id, text="Вот файл, где подробно указаны правила приема для поступления в КазУТБ на русском языке")

        # Отправка документа с правилами
        with open(document_path, 'rb') as document:
            logging.info("Отправка документа на русском языке")
            await context.bot.send_document(chat_id=chat_id, document=document)
    except Exception as e:
        logging.error(f"Ошибка при отправке документа: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при отправке документа. Пожалуйста, подождите пока файл отправится на сервер.")


# Функция для отправки PDF файла с правилами на казахском языке
async def send_rule_pdf_kaz(update: Update, context: CallbackContext) -> None:
    chat_id = update.message.chat_id
    document_path = 'rule_kaz.pdf'  # Путь к файлу с правилами

    try:
        logging.info("Отправка предварительного сообщения")
        await context.bot.send_message(chat_id=chat_id, text="Вот файл, где подробно указаны правила приема для поступления в КазУТБ на казахском языке")

        # Отправка документа с правилами
        with open(document_path, 'rb') as document:
            logging.info("Отправка документа на казахском языке")
            await context.bot.send_document(chat_id=chat_id, document=document)
    except Exception as e:
        logging.error(f"Ошибка при отправке документа: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка при отправке документа. Пожалуйста, подождите пока файл отправится на сервер.")

# Создание корпуса и векторизатора
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(data.keys())
similarity_matrix = cosine_similarity(X_train)

# Модель для генерации ответов
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(data), activation='softmax')
])

# Компиляция модели
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Подготовка входных данных для обучения
y_train = np.eye(len(data))[np.arange(len(data))]

# Обучение модели
model.fit(X_train.toarray(), y_train, epochs=100, verbose=1)

# Функция для предсказания ответа с использованием обученной модели
def predict_response(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    return list(data.values())[np.argmax(prediction)]

# Создание клавиатуры с кнопками
def get_keyboard():
    keyboard = [
        [KeyboardButton("Какие специальности есть?")],
        [KeyboardButton("Как поступить?")],
        [KeyboardButton("Есть ли общежитие?")],
        [KeyboardButton("Какие дополнительные услуги предоставляет университет?")],
        [KeyboardButton("Бакалавриат")],
        [KeyboardButton("Что я получу?")],
        [KeyboardButton("Информация о КазУТБ")],
        [KeyboardButton("Часто задаваемые вопросы")],
        [KeyboardButton("Почему КазУТБ?")],
        [KeyboardButton("Заявление на проживание в общежитии")],
        [KeyboardButton("Хочу пройти профориентацию")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Привет! Я бот-менеджер по консультированию абитурентов университета. Как я могу помочь вам?", reply_markup=get_keyboard())

# Обработчик команды /questions
async def questions(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Доступные вопросы боту:\n\n\n\n • Какие специальности есть?\n\n • Как поступить?\n\n • Есть ли общежитие?\n\n • Какие дополнительные услуги предоставляет университет?\n\n • Бакалавриат\n\n • Что я получу?\n\n • Информация о КазУТБ\n\n • Контакты приемной комиссии\n\n • Часто задаваемые вопросы\n\n • Почему мы, почему КазУТБ?\n\n • Заявление на проживание в общежитии\n\n •Хочу пройти профориентацию", reply_markup=get_keyboard())

# Обработчик входящего текстового сообщения
async def reply(update: Update, context: CallbackContext) -> None:
    user_input = update.message.text.lower()
    response = predict_response(user_input)
    await update.message.reply_text(response, reply_markup=get_keyboard())

def main() -> None:
    application = ApplicationBuilder().token("6819906126:AAEi5nxI1wBE69sxmc__o5oMUJqfxnmtEuU").build()

    application.add_handler(CommandHandler("start", start))  # Обработчик команды /start
    application.add_handler(CommandHandler("rule_rus", send_rule_pdf_rus))  # Обработчик команды /rule_rus
    application.add_handler(CommandHandler("rule_kaz", send_rule_pdf_kaz))  # Обработчик команды /rule_kaz
    application.add_handler(CommandHandler("questions", questions))  # Обработчик команды /questions
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, reply))

    application.run_polling()
if __name__ == '__main__':
    main()
