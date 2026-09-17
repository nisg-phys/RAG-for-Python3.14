
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv(override=True)

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
    # LLM CONFIG
    # -------------------------

    llm_provider: str = "openai"
    llm_model: str = "gpt-4.1"
    groq_llm_model: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_tokens: int = 512

    # -------------------------
    # RAG PARAMETERS
    # -------------------------

    chunk_size: int = 800
    chunk_overlap: int = 200
    top_k: int = 5
    retrieval_candidate_pool: int = 12
    use_query_rewriting: bool = False

    # AWS information
    aws_access_key_id: str = Field(validation_alias= "AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(validation_alias= "AWS_SECRET_ACCESS_KEY")
    aws_region: str = Field(validation_alias= "AWS_REGION")
    s3_bucket: str = Field(validation_alias= "S3_BUCKET")
    s3_chunks_key: str= Field(validation_alias= "S3_CHUNKS_KEY")
    s3_endpoint_url: str | None = Field(default=None, validation_alias="S3_ENDPOINT_URL")

    # observability
    observability_dir: str = Field(default="storage/observability", validation_alias="RAGBOT_OBSERVABILITY_DIR")
    log_level: str = Field(default="INFO", validation_alias="RAGBOT_LOG_LEVEL")
    log_json: bool = Field(default=False, validation_alias="RAGBOT_LOG_JSON")

    # -------------------------
    # OPIK TRACING
    # -------------------------
    # Opik itself reads OPIK_API_KEY / OPIK_WORKSPACE / OPIK_URL_OVERRIDE
    # straight from the environment, so only the app-level toggle and
    # project name live in Settings.
    opik_api_key: str = Field(validation_alias="OPIK_API_KEY")
    opik_project_name: str = Field(default="ragbot", validation_alias="OPIK_PROJECT_NAME")
    enable_opik_tracing: bool = Field(default=True, validation_alias="RAGBOT_ENABLE_OPIK")
    opik_workspace: str = Field(default="ragbot", validation_alias="OPIK_WORKSPACE")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
