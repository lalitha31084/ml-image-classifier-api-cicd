# Stage 1: Build
FROM python:3.9-slim-buster as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim-buster
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY models/ ./models/

ENV PATH=/root/.local/bin:$PATH
ENV MODEL_PATH=/app/models/my_classifier_model.h5

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]