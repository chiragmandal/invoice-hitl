FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata + source BEFORE installing
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src

ENV PYTHONPATH=/app/src

RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .

CMD ["python", "-m", "invoice_hitl.run"]
