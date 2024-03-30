FROM mcr.microsoft.com/devcontainers/python:3.10-buster

RUN apt-get update && apt-get -y install --no-install-recommends \
   python3-distutils python3-venv gcc \
   && apt-get autoremove -y && apt-get clean -y

RUN export PYTHON_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}.{2}".format(*version))'`
RUN echo "$PYTHON_VERSION"

COPY requirements.txt /tmp/
COPY Makefile /tmp/

ARG USER="vscode"
ARG VENV_PATH="/home/${USER}/venv"

RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
   && su $USER -c "${VENV_PATH}/bin/pip install --upgrade pip --no-cache-dir" \
   && su $USER -c "${VENV_PATH}/bin/pip install --no-cache-dir install -r /tmp/requirements.txt" \
   && rm -rf /tmp/requirements.txt 

RUN echo "requirements install done"

USER $USER

RUN echo "dockerbuild done"
