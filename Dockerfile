FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    # prevents python creating .pyc files
    PYTHONDONTWRITEBYTECODE=1 \
    \
    # pip
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    \
    # poetry
    # https://python-poetry.org/docs/configuration/#using-environment-variables
    POETRY_VERSION=1.7.1 \
    # make poetry install to this location
    POETRY_HOME="/opt/poetry" \
    # make poetry create the virtual environment in the project's root
    # it gets named `.venv`
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    # do not ask any interactive question
    POETRY_NO_INTERACTION=1 \
    \
    # paths
    # this is where our requirements + virtual environment will live
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv"
ENV POETRY_VIRTUALENVS_PATH = "$VENV_PATH"
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

ENV DEBIAN_FRONTEND=nonintercative
RUN apt-get update && apt-get install -y \
    curl \
    git \
    gcc \
    g++ \
    make \
    cmake \
    vim \
    wget \
    python3 \
    python3-distutils \
    zip\
    unzip \
    screen
#install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --git https://github.com/python-poetry/poetry.git@master
#ENV PATH="/root/.local/bin:${PATH}"
#RUN echo 'PATH="/root/.local/bin:${PATH}"' >> /root/.bashrc
#RUN source $HOME/.poetry/env && poetry update && poetry install
#RUN source /root/.bashrc
#ENV PATH="${PATH}:/root/.poetry/bin"
#RUN poetry config virtualenvs.in-project true
WORKDIR $PYSETUP_PATH
#COPY ./poetry.lock ./
#COPY ./pyproject.toml ./
COPY ./src ./src
COPY ./data ./data
COPY ./.env ./.env
COPY ./pyproject.toml ./pyproject.toml



#RUN eval "$(ssh-agent -s)"
#RUN ssh-add /home/vscode/.ssh/id_rsa_github
# RUN curl -fsSL https://get.docker.com | sh

RUN poetry install 

RUN poetry run pip install --upgrade pip

#RUN poetry init
#RUN poetry shell
