from aiogram import types

class CommandHandler:
    async def handle_start(self, message: types.Message):
        await message.answer(
            "Привет! Пришлите текст разговора для анализа тона и получения рекомендаций.\n"
            "Минимум 20 символов."
        )