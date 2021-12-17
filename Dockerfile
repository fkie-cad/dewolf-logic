FROM python:3.8

RUN mkdir -p /opt/logic
COPY . /opt/logic
WORKDIR /opt/logic

RUN pip install -r requirements.txt
