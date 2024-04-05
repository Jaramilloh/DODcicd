FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

RUN apt-get update && apt-get -y install --no-install-recommends \
   gcc ffmpeg libgl1 libsm6 libxext6 \
   python3-venv \
   && apt-get autoremove -y && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/*

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
