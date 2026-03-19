import os
from dotenv import load_dotenv
from ragbot.config.settings import settings

load_dotenv()

print("---- DEBUG ENV ----")
print("os.getenv:", os.getenv("GROQ_API_KEY"))
print("-------------------")

print("settings.groq_api_key:", settings.groq_api_key)