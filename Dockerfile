FROM python:3.9-slim

WORKDIR /app

COPY . /app/

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    && pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]