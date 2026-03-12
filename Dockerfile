FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Copy .env so it's available inside the container
COPY .env ./

WORKDIR /app/backend

EXPOSE 8000

# Both keys must be provided — either via .env or runtime env vars
ENV GROQ_API_KEY=""
ENV GROQ_API_KEY_WRITER=""
ENV PORT=8000

CMD ["python", "main.py"]
