import os
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from transformers import pipeline, AutoTokenizer, T5ForConditionalGeneration
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=os.getenv("TELEGRAM_TOKEN", "8114468392:AAF9d5Bq7coCZAhMYn9ZZwXzTm3vWz5Euxk"))
dp = Dispatcher()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", "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"),
)

logger.info("Загрузка моделей...")
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment"
)
logger.info("Модели загружены")

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Пришлите текст разговора для анализа тона и получения рекомендаций.\n"
        "Минимум 20 символов."
    )

async def generate_recommendations_with_deepseek(text: str, sentiment: str, score: float) -> str:
    if sentiment == "NEGATIVE":
        system_prompt = (
            "Вы - опытный бизнес-тренер, специализирующийся на работе с клиентами. "
            "Проанализируйте текст разговора и дайте 3-5 конкретных рекомендаций менеджеру "
            "по улучшению ситуации, используя профессиональные техники работы с возражениями."
        )
        user_prompt = (
            f"Текст разговора с клиентом (тональность: негативная, уверенность: {score:.2f}):\n\n"
            f"{text}\n\n"
            "Сформулируйте рекомендации в формате:\n"
            "1) [Конкретное действие] - [краткое объяснение]\n"
            "2) [Конкретное действие] - [краткое объяснение]\n"
            "..."
        )
    elif sentiment == "POSITIVE":
        system_prompt = (
            "Вы - опытный бизнес-тренер, специализирующийся на работе с клиентами. "
            "Проанализируйте текст разговора и дайте 3-5 конкретных рекомендаций менеджеру "
            "как усилить положительное впечатление и увеличить лояльность клиента."
        )
        user_prompt = (
            f"Текст разговора с клиентом (тональность: позитивная, уверенность: {score:.2f}):\n\n"
            f"{text}\n\n"
            "Сформулируйте рекомендации в формате:\n"
            "1) [Конкретное действие] - [краткое объяснение]\n"
            "2) [Конкретное действие] - [краткое объяснение]\n"
            "..."
        )
    else:
        system_prompt = (
            "Вы - опытный бизнес-тренер, специализирующийся на работе с клиентами. "
            "Проанализируйте текст разговора и дайте 3-5 конкретных рекомендаций менеджеру "
            "как установить более теплый контакт и перевести разговор в позитивное русло."
        )
        user_prompt = (
            f"Текст разговора с клиентом (тональность: нейтральная, уверенность: {score:.2f}):\n\n"
            f"{text}\n\n"
            "Сформулируйте рекомендации в формате:\n"
            "1) [Конкретное действие] - [краткое объяснение]\n"
            "2) [Конкретное действие] - [краткое объяснение]\n"
            "..."
        )
    
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            )
        
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"Ошибка при запросе к DeepSeek: {e}", exc_info=True)
        return "Не удалось сгенерировать рекомендации. Пожалуйста, попробуйте позже."

@dp.message()
async def analyze_conversation(message: types.Message):
    text = message.text
    
    if len(text) < 20:
        await message.answer("❌ Текст слишком короткий. Нужно минимум 20 символов.")
        return
    
    try:
        # Анализ тональности
        sentiment_result = sentiment_analyzer(text)[0]
        sentiment_label = sentiment_result["label"]
        score = sentiment_result["score"]
        
        tone = {
            "POSITIVE": "✅ Позитивный",
            "NEUTRAL": "🔸 Нейтральный",
            "NEGATIVE": "🔻 Негативный"
        }.get(sentiment_label, "🔸 Нейтральный")
        
        # Генерация рекомендаций через DeepSeek
        recommendations = await generate_recommendations_with_deepseek(text, sentiment_label, score)
        
        response = (
            f"📊 Тон разговора: {tone} (точность: {score:.2f})\n\n"
            f"💡 Рекомендации:\n{recommendations}"
        )
        
        await message.answer(response)
        
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
        await message.answer("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())