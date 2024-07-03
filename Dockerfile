FROM python:3.9-slim-buster

WORKDIR /python-docker

ENV PYTHONPATH /python-docker

# Update system and install dependencies
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*  # Clean up unnecessary files to reduce image size

COPY requirements.txt requirements.txt

COPY CEDAR /python-docker/CEDAR

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "endpoint/server.py", "-h", "0.0.0.0", "-p", "5000"]