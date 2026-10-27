"""
Unit test offline (tanpa memanggil Gemini API): tool, ekstraksi ticket_id, router, dan
penanganan error endpoint /ask. Pemanggilan LLM diganti dengan fungsi palsu via monkeypatch.

Jalankan: pytest
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app import router
from app.main import app
from app.rag import LLMError
from app.tools import find_ticket, get_ticket_status

# TestClient tanpa `with` tidak menjalankan lifespan, sehingga index FAISS tidak dibangun
client = TestClient(app, raise_server_exceptions=False)


def fake_llm_intent(intent):
    async def _fake(query):
        return intent
    return _fake


async def failing_llm(*args, **kwargs):
    raise LLMError("simulasi Gemini API down")


# ---------- tools.py ----------

def test_ticket_found():
    message = get_ticket_status("TICKET-0005")
    assert "Diproses" in message and "Reset password VPN gagal terus" in message


def test_ticket_lookup_is_case_insensitive():
    assert find_ticket("ticket-0005")["status"] == "Diproses"


def test_ticket_not_found_does_not_raise():
    assert find_ticket("TICKET-0042") is None
    assert "tidak ditemukan" in get_ticket_status("TICKET-0042")


def test_employee_name_is_not_exposed():
    assert "employee_name" not in find_ticket("TICKET-0001")


# ---------- router.py: ekstraksi ticket_id ----------

@pytest.mark.parametrize("query, expected", [
    ("Apa status tiket TICKET-0042 saya?", "TICKET-0042"),
    ("cek ticket-5 dong", "TICKET-0005"),
    ("TIKET0012 sudah selesai belum?", "TICKET-0012"),
    ("Tiket saya nomor 7 sudah sampai mana?", "TICKET-0007"),
    ("What's the status of ticket 0003?", "TICKET-0003"),
    ("ticket #12 gimana kabarnya", "TICKET-0012"),
    ("tiket saya belum ada kabar", None),
    ("tiket yang saya buat 2 hari lalu belum diproses", None),
])
def test_extract_ticket_id(query, expected):
    assert router.extract_ticket_id(query) == expected


# ---------- router.py: intent detection ----------

@pytest.mark.parametrize("query, expected", [
    ("Apa status tiket TICKET-0005?", "ticket_status"),
    ("tolong cek tiket saya", "ticket_status"),
    ("Berapa jatah cuti tahunan?", "faq"),
    ("Bagaimana cara membuat tiket IT?", "faq"),
])
def test_keyword_baseline(query, expected):
    assert router.keyword_intent(query) == expected


def test_explicit_ticket_id_skips_llm(monkeypatch):
    monkeypatch.setattr(router, "llm_intent", failing_llm)
    assert asyncio.run(router.detect_intent("cek TICKET-0001")) == ("ticket_status", "rule")


def test_llm_failure_falls_back_to_keyword(monkeypatch):
    monkeypatch.setattr(router, "generate", failing_llm)
    assert asyncio.run(router.detect_intent("status tiket saya gimana?")) == ("ticket_status", "keyword_fallback")
    assert asyncio.run(router.detect_intent("berapa jatah cuti?")) == ("faq", "keyword_fallback")


def test_invalid_llm_label_falls_back_to_keyword(monkeypatch):
    async def bad_label(*args, **kwargs):
        return "billing"
    monkeypatch.setattr(router, "generate", bad_label)
    assert asyncio.run(router.detect_intent("berapa jatah cuti?")) == ("faq", "keyword_fallback")


# ---------- main.py: endpoint /ask ----------

def test_ask_ticket_found():
    response = client.post("/ask", json={"query": "Apa status tiket TICKET-0005 saya?"})
    body = response.json()
    assert response.status_code == 200
    assert (body["status"], body["intent"]) == ("success", "ticket_status")
    assert body["metadata"]["ticket"]["status"] == "Diproses"


def test_ask_ticket_not_found():
    body = client.post("/ask", json={"query": "cek TICKET-0042"}).json()
    assert (body["status"], body["intent"]) == ("not_found", "ticket_status")


def test_ask_ticket_status_without_id(monkeypatch):
    monkeypatch.setattr(router, "llm_intent", fake_llm_intent("ticket_status"))
    body = client.post("/ask", json={"query": "tiket saya sudah sampai mana?"}).json()
    assert (body["status"], body["intent"]) == ("error", "ticket_status")
    assert "TICKET-XXXX" in body["data"]


def test_ask_faq_routes_to_rag(monkeypatch):
    async def fake_rag(query):
        return {"answer": "12 hari.\nSumber: FAQ #1", "sources": [], "cited_faq_ids": [1]}
    monkeypatch.setattr(router, "llm_intent", fake_llm_intent("faq"))
    monkeypatch.setattr(router, "answer_from_knowledge_base", fake_rag)
    body = client.post("/ask", json={"query": "berapa jatah cuti?"}).json()
    assert (body["status"], body["intent"], body["data"]) == ("success", "faq", "12 hari.\nSumber: FAQ #1")


def test_ask_unknown_intent(monkeypatch):
    monkeypatch.setattr(router, "llm_intent", fake_llm_intent("unknown"))
    response = client.post("/ask", json={"query": "siapa juara piala dunia?"})
    assert response.status_code == 200
    assert (response.json()["status"], response.json()["intent"]) == ("error", "unknown")


def test_ask_llm_down_returns_503(monkeypatch):
    monkeypatch.setattr(router, "llm_intent", fake_llm_intent("faq"))
    monkeypatch.setattr(router, "answer_from_knowledge_base", failing_llm)
    response = client.post("/ask", json={"query": "berapa jatah cuti?"})
    assert response.status_code == 503
    assert (response.json()["status"], response.json()["intent"]) == ("error", "faq")


def test_ask_ticket_path_survives_llm_outage(monkeypatch):
    monkeypatch.setattr(router, "generate", failing_llm)
    body = client.post("/ask", json={"query": "status tiket saya nomor 9?"}).json()
    assert (body["status"], body["metadata"]["router"]) == ("success", "keyword_fallback")


@pytest.mark.parametrize("payload", [{"query": "   "}, {}, {"query": "x" * 1001}])
def test_ask_invalid_request_returns_structured_422(payload):
    response = client.post("/ask", json=payload)
    assert response.status_code == 422
    assert set(response.json()) == {"status", "intent", "data", "metadata"}


def test_ask_unexpected_error_returns_structured_500(monkeypatch):
    async def crash(query):
        raise ValueError("bug")
    monkeypatch.setattr("app.main.handle_query", crash)
    response = client.post("/ask", json={"query": "halo"})
    assert response.status_code == 500
    assert response.json()["status"] == "error"
