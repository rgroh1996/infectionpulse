FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-service.txt requirements-lock.txt ./
RUN pip install --no-cache-dir -c requirements-lock.txt -r requirements-service.txt
COPY infectionpulse ./infectionpulse
COPY examples ./examples
COPY scripts/serve.py ./scripts/
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8000"]
