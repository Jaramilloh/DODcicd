# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/codespaces-linux/.devcontainer/base.Dockerfile

#FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
#RUN apt-get update && apt-get -y install --no-install-recommends \
   #python3-distutils \
   #python3-venv \
   #gcc 

FROM mcr.microsoft.com/devcontainers/python:3.10-buster

RUN apt-get update && apt-get -y install --no-install-recommends \
   python3-distutils python3-venv gcc \
   && apt-get autoremove -y && apt-get clean -y

RUN export PYTHON_VERSION=`python3 -c 'import sys; version=sys.version_info[:3]; print("{0}.{1}.{2}".format(*version))'`
RUN echo "$PYTHON_VERSION"

ARG USER="vscode"
ARG VENV_PATH="/home/${USER}/venv"
COPY requirements.txt /tmp/
COPY Makefile /tmp/
RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
   && su $USER -c "${VENV_PATH}/bin/pip install --upgrade pip --no-cache-dir" \
   && su $USER -c "${VENV_PATH}/bin/pip install --no-cache-dir install -r /tmp/requirements.txt" \
   && rm -rf /tmp/requirements.txt 
