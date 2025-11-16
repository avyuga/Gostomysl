import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # YandexGPT settings
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
    YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
    YANDEX_GPT_MODEL = "yandexgpt-lite"
    YANDEX_GPT_MODEL_URI = f"gpt://{YANDEX_FOLDER_ID}/{YANDEX_GPT_MODEL}"
    
    # ArXiv settings
    ARXIV_MAX_RESULTS = 100
    TOP_K_BM25 = 50
    TOP_K_EMBEDDING = 25
    TOP_K_FINAL = 10
    
    # Redis settings (for caching)
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
