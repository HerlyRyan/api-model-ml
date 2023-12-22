FROM python:3.10.3-slim-buster

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "recommendation:app", "--host", "0.0.0.0", "--port", "8080"]
