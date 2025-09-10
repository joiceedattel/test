from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = Field("assistant", env="APP_NAME")

    # Redis settings
    redis_sentinel_host: str = Field("rfs-assistant", env="REDIS_SENTINEL_HOST")
    redis_sentinel_port: int = Field(26379, env="REDIS_SENTINEL_PORT")
    redis_sentinel_master: str = Field("mymaster", env="REDIS_SENTINEL_MASTER")
    redis_password: str | None = Field(None, env="REDIS_PASSWORD")
    redis_db: int = Field(0, env="REDIS_DB")

    api_base_url: str = Field("https://dev-api.item24.com/", env="API_BASE_URL")

    keycloak_issuer: str
    keycloak_client_id: str

    docs_url: str | None = None

    knowledge_graph_api_url: str = (Field(env="KNOWLEDGE_GRAPH_API_URL"),)

    max_conversation_turns: int = Field(5, env="MAX_CONVERSATION_TURNS")

    translator_key: str
    translator_location: str
    translator_category_id: str
    
    guardrail_enabled: bool = Field(False, env="GUARDRAIL_ENABLED")
    guardrail_openai_key : str
    guardrail_azure_endpoint : str
    guardrail_api_version : str
    guardrail_model : str

    # Database settings
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")
    db_host: str = Field(..., env="DB_HOST")
    db_port: int = Field(3306, env="DB_PORT")
    db_name: str = Field(..., env="DB_NAME")

    class Config:
        env_file = ".env"  # Loads from .env automatically
        extra = "allow"  # Allows extra fields not defined in the model


settings = Settings()

FAITHFULNESS_THRESHOLD = 0.8
RELEVANCE_THRESHOLD = 0.8
