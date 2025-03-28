FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl build-essential

ENV POETRY_VERSION=1.5.1
RUN curl -sSL https://install.python-poetry.org | python - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
 && poetry install --no-dev --no-interaction

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
