# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/codespaces-linux/.devcontainer/base.Dockerfile

#FROM mcr.microsoft.com/devcontainers/base:ubuntu-20.04

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
#RUN apt-get update && apt-get -y install --no-install-recommends \
   #python3-distutils \
   #python3-venv \
   #gcc 

FROM mcr.microsoft.com/devcontainers/python:3.9-buster

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
   && apt-get -y install clang lld lshw python3.9-venv python3.9-dev libssl-dev \
   && apt-get autoremove -y && apt-get clean -y
   
ARG USER="vscode"
ARG VENV_PATH="/home/${USER}/venv"
COPY requirements.txt /tmp/
COPY Makefile /tmp/
RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
   && su $USER -c "${VENV_PATH}/bin/pip --disable-pip-version-check --no-cache-dir install -r /tmp/requirements.txt" \
   && rm -rf /tmp/requirements.txt 
