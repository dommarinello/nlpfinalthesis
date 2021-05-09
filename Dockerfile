FROM python:3

WORKDIR /app

COPY app.py app.py
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "app.py"]
