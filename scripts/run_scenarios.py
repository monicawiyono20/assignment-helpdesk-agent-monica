"""
Uji end-to-end endpoint /ask: 5 skenario wajib + 3 skenario tambahan.
Server (lokal maupun container Docker) harus sudah berjalan. Hanya memakai standard library.

Jalankan: python scripts/run_scenarios.py [BASE_URL]      (default: http://localhost:8000)
Hasil disimpan ke results/scenario_results.json
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "scenario_results.json"

SCENARIOS = [
    # (id, jenis, query, intent yang diharapkan, status yang diharapkan)
    ("S1", "RAG", "Berapa jatah cuti tahunan saya sebagai karyawan tetap?", "faq", "success"),
    ("S2", "RAG", "Gimana caranya reset password VPN? Akun saya kayaknya kekunci", "faq", "success"),
    ("S3", "Tool call", "Apa status tiket TICKET-0005 saya?", "ticket_status", "success"),
    ("S4", "Tool call (ID tidak ada)", "Tolong cek status tiket TICKET-0042", "ticket_status", "not_found"),
    ("S5", "Di luar cakupan KB", "Apakah kantor menanggung biaya kacamata karyawan?", "faq", "success"),
    # Skenario tambahan untuk kasus tepi router
    ("S6", "Tambahan: kata 'tiket' di pertanyaan FAQ", "Saya harus bikin tiket nggak kalau mau minta laptop baru?", "faq", "success"),
    ("S7", "Tambahan: cek tiket tanpa ID", "Laporan laptop overheat yang saya kirim minggu lalu sudah ada update belum?", "ticket_status", "error"),
    ("S8", "Tambahan: di luar domain", "Siapa pemenang Piala Dunia 2022?", "unknown", "error"),
]


def ask(query: str) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{BASE_URL}/ask",
        data=json.dumps({"query": query}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:  # 4xx/5xx tetap berisi body terstruktur
        return exc.code, json.load(exc)


def main() -> None:
    results = []
    for scenario_id, kind, query, expected_intent, expected_status in SCENARIOS:
        start = time.perf_counter()
        http_status, body = ask(query)
        latency_ms = round((time.perf_counter() - start) * 1000)
        results.append({
            "id": scenario_id,
            "jenis": kind,
            "query": query,
            "expected_intent": expected_intent,
            "expected_status": expected_status,
            "http_status": http_status,
            "latency_ms": latency_ms,
            "routed_correctly": body["intent"] == expected_intent,
            "status_as_expected": body["status"] == expected_status,
            "response": body,
        })
        mark = "OK  " if body["intent"] == expected_intent else "FAIL"
        print(f"[{mark}] {scenario_id} HTTP {http_status} {latency_ms:>5} ms  intent={body['intent']:<13} "
              f"status={body['status']:<9} router={body['metadata'].get('router')}")
        print(f"       Q: {query}")
        print(f"       A: {body['data']}\n")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    routed = sum(r["routed_correctly"] for r in results)
    print(f"Routing benar: {routed}/{len(results)}. Hasil lengkap: {OUTPUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
