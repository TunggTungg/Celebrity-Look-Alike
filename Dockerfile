# Stage 1: Build stage
FROM python:3.10-slim AS build-web-stage

# Install required packages using apt
RUN apt-get update && \
    apt-get install -y libgl1 curl && \
    rm -rf /var/lib/apt/lists/*
# Install UV
ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh
COPY ./requirements/requirements.txt .
RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements.txt

# Stage 2: web stage
FROM python:3.10-slim as web-final 

# Install libgl1 from the build stage
COPY --from=build-web-stage /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
# Copy built Python libraries from the build stage
COPY --from=build-web-stage /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=build-web-stage /usr/local/bin/uvicorn /usr/local/bin/uvicorn
# Set the working directory
WORKDIR /app

# # Stage 3: Build stage
# FROM python:3.10-slim AS build-server-stage
# # Install UV from the build stage
# COPY --from=build-web-stage /root/.cargo/bin/uv /root/.cargo/bin/uv
# COPY ./requirements/requirements-tritonserver.txt .
# RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements-tritonserver.txt

# Stage 4: triton server
FROM nvcr.io/nvidia/tritonserver:24.01-py3 as tritonserver-final
# Install libgl1 from the build stage
# COPY --from=build-web-stage /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu
# COPY --from=build-server-stage /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# Install dependencies
RUN apt update && \
    apt install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=build-web-stage /root/.cargo/bin/uv /root/.cargo/bin/uv
COPY ./requirements/requirements-tritonserver.txt .
RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements-tritonserver.txt

 
