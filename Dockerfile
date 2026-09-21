FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn ragbot.main:app --host 0.0.0.0 --port ${PORT}"]
