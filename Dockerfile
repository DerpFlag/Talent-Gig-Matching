# Hugging Face Docker Space + local `docker build` entry for the Streamlit product UI.
# HF exposes port 7860 by default (see README.md YAML `app_port`).
# For API-only image use Dockerfile.api (see docker-compose.yml).

FROM python:3.11-slim

RUN useradd -m -u 1000 user

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 7860

CMD ["streamlit", "run", "src/ui/product_app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true", "--browser.gatherUsageStats", "false"]
