FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for Hugging Face Spaces compatibility (UID 1000)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy repository source code, datasets, and static vector stores
COPY --chown=user:user . /app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["streamlit", "run", "src/unified_main.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
