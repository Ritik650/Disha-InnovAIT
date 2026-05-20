FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY disha/ ./disha/
RUN pip install --no-cache-dir -e .

# data/ is mounted as a read-only volume by docker-compose so judges can
# point Disha at fresh artifacts without rebuilding the image.
EXPOSE 8000
CMD ["uvicorn", "disha.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
