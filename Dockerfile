FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y python3-pip python3.10 python3-libnvinfer libgl1 libglib2.0-0 libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/* \
    && cd /usr/local/bin \
    && ln -s /usr/bin/python3 python \
    && pip3 install --upgrade pip

RUN mkdir /app
WORKDIR /app
COPY requirements/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt 



