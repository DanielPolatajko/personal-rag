FROM python:3.13-slim as base

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

COPY requirements.txt .

RUN uv pip install -r requirements.txt --system

COPY personal_rag/ personal_rag/

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["streamlit", "run", "personal_rag/ui/app.py", "--server.address", "0.0.0.0"]

FROM base as dev

COPY .vscode/ .vscode/
COPY .devcontainer/ .devcontainer/
COPY README.md .
COPY docker-compose.yml .
COPY .gitignore .

RUN apt-get update && apt-get install -y git apt-transport-https ca-certificates gnupg curl
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt-get update -y && apt-get install google-cloud-cli -y
    
