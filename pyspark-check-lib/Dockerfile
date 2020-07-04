FROM python:3.8.1-slim


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
    POETRY_VERSION=1.0.3 \
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


# prepend poetry and venv to path
ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for installing poetry
        curl \
        wget \
        # deps for building python deps
        build-essential

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

RUN cd /opt && wget -c --header "Cookie:oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u131-b11/d54c1d3a095b4ff2b6607d096fa80163/jdk-8u131-linux-x64.tar.gz
RUN tar -xzf /opt/jdk-8u131-linux-x64.tar.gz -C /opt && ln -s /opt/jdk1.8.0_131 /opt/jdk

ENV JAVA_HOME /opt/jdk
ENV PATH ${PATH}:${JAVA_HOME}/bin

RUN mkdir -p /opt/spark \
    && cd /opt/spark \
    && curl http://ftp.man.poznan.pl/apache/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz -o spark-2.4.6-bin-hadoop2.7.tgz \
    && tar xf spark-2.4.6-bin-hadoop2.7.tgz

WORKDIR /app
COPY pyspark_check ./pyspark_check
COPY ./tests ./tests
COPY ./pyproject.toml poetry.lock ./
COPY ./conftest.py ./
COPY ./README.md ./

RUN poetry install

ENV SPARK_HOME=/opt/spark/spark-3.0.0-bin-hadoop2.7

CMD poetry run pytest -s --cov=pyspark_check --cov-branch --cov-fail-under=80 tests/