
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    # -------------------------
    # API KEYS
    # -------------------------

    openai_api_key: str = Field(validation_alias= "OPENAI_API_KEY")
    groq_api_key: str = Field(validation_alias= "GROQ_API_KEY")
    pinecone_api_key: str = Field(validation_alias="PINECONE_API_KEY")

    # -------------------------
    # PINECONE CONFIG
    # -------------------------

    pinecone_index_name: str = Field(default="ragbot-index", validation_alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(validation_alias="PINECONE_ENVIRONMENT")

    # -------------------------
    # EMBEDDING MODEL
    # -------------------------

    embedding_model: str = "text-embedding-3-small"

    # -------------------------
    # LLM CONFIG (Groq)
    # -------------------------

    llm_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 512

    # -------------------------
    # RAG PARAMETERS
    # -------------------------

    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 5

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
