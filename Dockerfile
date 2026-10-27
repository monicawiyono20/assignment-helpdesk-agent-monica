# Image untuk IT Helpdesk Assistant API, mengikuti pola Sesi 14.2:
# FROM python -> WORKDIR -> COPY requirements -> RUN pip -> COPY source -> EXPOSE -> CMD
#
# API key TIDAK disimpan di image. Berikan saat menjalankan container:
#   docker run -p 8000:8000 --env-file .env it-helpdesk-assistant

FROM python:3.11-slim

# Tanpa file .pyc dan tanpa buffering output, agar log langsung terlihat di `docker logs`
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

# requirements.txt di-copy dan di-install lebih dulu agar layer dependency di-cache:
# perubahan source code tidak memicu install ulang dependency
COPY requirements.txt .
RUN pip install -r requirements.txt

# Hanya source code dan data yang dibutuhkan saat runtime (.env dikecualikan lewat .dockerignore)
COPY app/ ./app/
COPY data/ ./data/

# Jalankan aplikasi sebagai user non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# --host 0.0.0.0 wajib agar endpoint dapat diakses dari luar container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
