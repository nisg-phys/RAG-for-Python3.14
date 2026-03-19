from fastapi import FastAPI
from ragbot.api.routes import router


app = FastAPI(
    title="RAG Bot",
    version="1.0"
)

app.include_router(router)