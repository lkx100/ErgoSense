FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libglib2.0-0 build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml uv.lock* /app/

# Install pip + uv; sync dependencies (creates .venv)
RUN pip install --upgrade pip && pip install uv
RUN uv sync --frozen --locked || uv sync

# Copy app code after deps to maximize cache
COPY . /app

# Non-root user (best practice)
RUN useradd -m ergo && chown -R ergo:ergo /app
USER ergo

EXPOSE 8501
ENV PYTHONPATH=/app
CMD [".venv/bin/uv", "run", "streamlit", "run", "ergo_web.py", "--server.port=8501", "--server.address=0.0.0.0"]
