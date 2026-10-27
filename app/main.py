"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client

Seluruh response /ask, termasuk saat error, memakai struktur yang sama: status, intent, data, metadata.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv

# Dimuat sebelum import modul app, karena konfigurasi model dibaca dari environment saat import.
# Di Docker tidak ada file .env; variabel diberikan lewat `docker run --env-file .env`.
load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from app import rag, tools  # noqa: E402
from app.rag import LLMError  # noqa: E402
from app.router import handle_query  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
# Sembunyikan log SDK tentang automatic function calling (fitur itu tidak dipakai di sini)
logging.getLogger("google_genai.models").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

LLM_UNAVAILABLE_MESSAGE = "Maaf, layanan AI sedang tidak dapat merespons. Silakan coba lagi beberapa saat lagi."


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Konfigurasi salah (API key kosong, data hilang) -> gagal saat startup dengan pesan jelas
    rag.get_client()
    tools.load_tickets()
    # Gangguan sementara pada Gemini API -> aplikasi tetap jalan; jalur cek tiket tidak butuh index,
    # dan index dicoba dibangun ulang pada request FAQ berikutnya
    try:
        await rag.build_index()
    except LLMError as exc:
        logger.warning("Index FAISS gagal dibangun saat startup, akan dicoba lagi saat request FAQ: %s", exc)
    yield


app = FastAPI(title="IT Helpdesk Assistant API", lifespan=lifespan)


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=1000, examples=["berapa jatah cuti tahunan saya?"])


class AskResponse(BaseModel):
    status: Literal["success", "not_found", "error"]
    intent: Literal["faq", "ticket_status", "unknown"]
    data: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def error_response(http_status: int, intent: str, message: str) -> JSONResponse:
    body = AskResponse(status="error", intent=intent, data=message)
    return JSONResponse(status_code=http_status, content=body.model_dump())


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    # Intent detection punya fallback keyword, jadi LLMError yang sampai ke sini selalu dari jalur RAG
    logger.error("LLM gagal merespons pada jalur RAG: %s", exc)
    return error_response(503, "faq", LLM_UNAVAILABLE_MESSAGE)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return error_response(422, "unknown", 'Request tidak valid: kirim JSON {"query": "..."} berisi teks 1-1000 karakter.')


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    # Traceback tetap dicatat oleh uvicorn; client hanya menerima pesan generik
    return error_response(500, "unknown", "Maaf, terjadi kesalahan internal. Silakan coba lagi.")


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {
        "status": "ok",
        "faq_indexed": rag.index_size(),
        "llm_model": rag.LLM_MODEL,
        "embedding_model": rag.EMBEDDING_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    Penanganan error:
      - intent tidak dikenali        -> 200, status "error", intent "unknown"
      - ticket_id tidak ada di data  -> 200, status "not_found"
      - LLM gagal merespons (RAG)    -> 503, status "error"
      - body request tidak valid     -> 422, status "error"
      - error tak terduga            -> 500, status "error"
    """
    result = await handle_query(request.query)
    logger.info("intent=%s router=%s status=%s", result["intent"], result["metadata"].get("router"), result["status"])
    return AskResponse(**result)
