
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):

    API_URL: str = "http://api:8000"

    model_config = SettingsConfigDict(env_file=".env")

config = AppConfig()