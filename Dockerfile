ARG IMAGE_TAG

# ------ Base -----
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime as setup
ARG ARTIFACTORY_LOGIN
ARG ARTIFACTORY_TOKEN

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PYTHONHASHSEED=random \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100 \
  POETRY_VERSION=1.1.7 \
  POETRY_HOME="/usr/local/poetry"

# We need to remove PyYAML dist-util.
RUN conda remove PyYAML -y

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y curl git && \
  rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python \
  && ln -sf /usr/local/poetry/bin/poetry /usr/local/bin/poetry

# Install dependencies.
COPY poetry.lock pyproject.toml /app/

WORKDIR /app
RUN poetry config virtualenvs.create false && \
  poetry install --no-interaction --no-ansi --no-root --no-dev

# Install the project.
COPY . /app/
RUN poetry install --no-interaction --no-ansi --no-dev

# ---- test -----
FROM setup as test_baal

WORKDIR /app/baal

RUN poetry install --no-interaction --no-ansi

# ---- release image ----
FROM setup as release