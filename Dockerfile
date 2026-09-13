FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN addgroup --system orchestrator && adduser --system --ingroup orchestrator orchestrator
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /data /app/workspace /app/artifacts && chown -R orchestrator:orchestrator /data /app

USER orchestrator
EXPOSE 8000
CMD ["python", "-m", "orchestrator.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]
