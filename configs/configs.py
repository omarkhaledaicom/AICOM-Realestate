import os
from dotenv import load_dotenv


load_dotenv()


class Configs:

    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PROXY = os.getenv("PROXY")
    AI_MODEL = os.getenv("MODEL", "gpt-4.1-mini")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", 1024))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
    TOP_P = float(os.getenv("TOP_P", 1.0))

    MAX_HISTORY_LENGTH = int(os.getenv("MAX_HISTORY_LENGTH", 10))
    DB_NAME = os.getenv("DB_NAME", "real_estate_db")
    DB_USER = os.getenv("DB_USER", "your_username")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "your_password")
    DB_PORT = int(os.getenv("DB_PORT", 5432))

    MAX_RESULTS_FOR_SUMMARY = int(os.getenv("MAX_RESULTS_FOR_SUMMARY", 3))
    MAX_RESULTS_FOR_BIOLINK = int(os.getenv("MAX_RESULTS_FOR_BIOLINK", 10))
    
