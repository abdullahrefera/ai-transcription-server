import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL: str = "gpt-4.1-mini"  # Only use GPT-4.1-mini
    
    # Performance optimization settings for sub-10-second responses
    MAX_TOKENS: int = 4000  # Increased for complex JSON responses
    TEMPERATURE: float = 0.2  # Low temperature for speed
    REQUEST_TIMEOUT: int = 60  # 60 seconds timeout

settings = Settings()
