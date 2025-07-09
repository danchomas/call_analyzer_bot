import logging
from aiogram import types
from transformers import pipeline
from openai import OpenAI
from constants.prompts import SYSTEM_PROMPTS, USER_PROMPT_TEMPLATE

class MessageHandler:
    def __init__(self, sentiment_analyzer: pipeline, openai_client: OpenAI):
        self.sentiment_analyzer = sentiment_analyzer
        self.openai_client = openai_client
        self.logger = logging.getLogger(__name__)

    async def handle_message(self, message: types.Message):
        text = message.text
        
        if len(text) < 20:
            await message.answer("❌ Текст слишком короткий. Нужно минимум 20 символов.")
            return
        
        try:
            sentiment_result = self.sentiment_analyzer(text)[0]
            sentiment_label = sentiment_result["label"]
            score = sentiment_result["score"]
            
            tone = {
                "POSITIVE": "✅ Позитивный",
                "NEUTRAL": "🔸 Нейтральный",
                "NEGATIVE": "🔻 Негативный"
            }.get(sentiment_label, "🔸 Нейтральный")
            
            recommendations = await self._generate_recommendations(text, sentiment_label, score)
            
            response = (
                f"📊 Тон разговора: {tone} (точность: {score:.2f})\n\n"
                f"💡 Рекомендации:\n{recommendations}"
            )
            
            await message.answer(response)
            
        except Exception as e:
            self.logger.error(f"Ошибка: {e}", exc_info=True)
            await message.answer("⚠️ Произошла ошибка при обработке запроса. Попробуйте позже.")

    async def _generate_recommendations(self, text: str, sentiment: str, score: float) -> str:
        system_prompt = SYSTEM_PROMPTS.get(sentiment, SYSTEM_PROMPTS["NEUTRAL"])
        user_prompt = USER_PROMPT_TEMPLATE.format(
            sentiment=sentiment.lower(), 
            score=score, 
            text=text
        )
        
        try:
            completion = self.openai_client.chat.completions.create(
                model="openai/o1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,  # Добавляем креативности
                max_tokens=600,   # Увеличиваем лимит токенов
            )
            return self._format_response(completion.choices[0].message.content.strip())
        
        except Exception as e:
            self.logger.error(f"Ошибка при запросе к DeepSeek: {e}", exc_info=True)
            return "Не удалось сгенерировать рекомендации. Пожалуйста, попробуйте позже."

    def _format_response(self, text: str) -> str:
        """Дополнительное форматирование ответа"""
        return text.replace("**", "*")