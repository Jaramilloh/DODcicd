# Use the Microsoft Python 3.9 base image
FROM mcr.microsoft.com/vscode/devcontainers/python:3.9

# Install necessary libraries
RUN apt-get update && apt-get install -y \
   gcc \
   g++ \
   make \
   libffi-dev \
   libssl-dev \
   ffmpeg libgl1 libsm6 libxext6 \
   python3-dev \
   python3-venv \
   libpng-dev \
   libfreetype6-dev \
   libblas-dev \
   && apt-get autoremove -y && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/
COPY Makefile /tmp/

ARG USER="vscode"

ARG VENV_PATH="/home/${USER}/venv"

ARG mkdir /home/${USER}/venv/tmp

RUN su $USER -c "/usr/bin/python3 -m venv /home/${USER}/venv" \
   && su $USER -c "TMPDIR=/home/${USER}/venv/tmp ${VENV_PATH}/bin/pip install --upgrade pip --no-cache-dir" \
   && su $USER -c "TMPDIR=/home/${USER}/venv/tmp ${VENV_PATH}/bin/pip install --no-cache-dir install -r /tmp/requirements.txt" \
   && rm -rf /tmp/requirements.txt 

RUN echo "requirements install done"

USER $USER
