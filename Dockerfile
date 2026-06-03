FROM python:3.11-slim

WORKDIR /app

# Install dependencies first — separate layer so it is cached on rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the running app needs
COPY model/      ./model/
COPY templates/  ./templates/
COPY app.py      .

EXPOSE 5000

CMD ["python", "app.py"]
