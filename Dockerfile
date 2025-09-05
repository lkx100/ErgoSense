FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv; sync dependencies (creates .venv)
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh

WORKDIR /app

# Copy app code after deps to maximize cache
# COPY pyproject.toml uv.lock* /app/
COPY . /app
RUN /usr/local/bin/uv sync

# Non-root user (best practice)
RUN useradd -m ergo && chown -R ergo:ergo /app
USER ergo

EXPOSE 8501
ENV PYTHONPATH=/app
CMD ["/usr/local/bin/uv", "run", "streamlit", "run", "ergo_web.py", "--server.port=8501", "--server.address=0.0.0.0"]
