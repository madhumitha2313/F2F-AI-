# F2F AI — Failure Intelligence Engine API
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pieces the API actually needs at runtime
COPY api/app.py ./api/app.py
COPY api/templates/ ./api/templates/
COPY api/static/ ./api/static/
COPY src/ ./src/
COPY monitoring/ ./monitoring/
COPY models/ ./models/

# monitoring/predictions.db is created on first write if it doesn't exist;
# mount a volume here in production so logs survive container restarts.
VOLUME ["/app/monitoring"]

EXPOSE 5000

ENV PORT=5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--chdir", "api", "app:app"]
