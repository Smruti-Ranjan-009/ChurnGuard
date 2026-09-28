FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

COPY requirements.txt setup.py ./
RUN pip install --no-cache-dir .

COPY app.py ./
COPY src/ ./src/
COPY templates/ ./templates/
COPY artifacts/model.pkl artifacts/preprocessor.pkl ./artifacts/

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=2)" || exit 1

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --access-logfile - --error-logfile - app:app"]