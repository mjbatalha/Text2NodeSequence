FROM python:3.12.3-slim
WORKDIR /app
COPY . .
RUN python3 -m venv venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt
RUN python metrics.py
CMD ["python", "main.py"]

