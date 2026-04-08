FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY middleware.py .

EXPOSE 4001

CMD ["python", "middleware.py"]
