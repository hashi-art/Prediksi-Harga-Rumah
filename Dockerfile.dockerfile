FROM python:3.11-slim

WORKDIR /app

# Install build deps then runtime deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app app
COPY web web
# Note: models/ directory not included by default. If you have trained model, copy it into models/ before building,
# or mount it at runtime: docker run -v $(pwd)/models:/app/models ...
COPY train.py .

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]