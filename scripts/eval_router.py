"""
Evaluasi akurasi router: keyword baseline vs router final (rule -> LLM -> fallback keyword)
pada query berlabel, termasuk ketepatan ekstraksi ticket_id. Memanggil Gemini API (butuh .env).

Jalankan dari root project: python -m scripts.eval_router
Hasil disimpan ke results/router_eval.csv
"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd  # noqa: E402

from app import router  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "results" / "router_eval.csv"

# (query, intent yang benar, ticket_id yang benar untuk query ticket_status)
LABELED_QUERIES = [
    ("Berapa hari jatah cuti tahunan untuk karyawan tetap?", "faq", None),
    ("gmn cara ganti rekening buat gajian?", "faq", None),
    ("Apakah sisa cuti tahun ini bisa dipakai tahun depan?", "faq", None),
    ("Can I work from home a few days a week?", "faq", None),
    ("Saya harus bikin tiket nggak kalau mau minta laptop baru?", "faq", None),
    ("Kalau laptop rusak, kategori apa yang harus saya pilih di tiket saya?", "faq", None),
    ("Should I create a ticket to get access to a new internal system?", "faq", None),
    ("cek TICKET-0011 dong", "ticket_status", "TICKET-0011"),
    ("ticket-12 update terbarunya apa?", "ticket_status", "TICKET-0012"),
    ("Tiket saya nomor 7 sudah sampai mana?", "ticket_status", "TICKET-0007"),
    ("What's the status of ticket 0003?", "ticket_status", "TICKET-0003"),
    ("Laporan laptop overheat yang saya kirim minggu lalu sudah ada update belum?", "ticket_status", None),
    ("Halo, selamat pagi!", "unknown", None),
    ("Siapa pemenang Piala Dunia 2022?", "unknown", None),
    ("Buatkan resep nasi goreng dong", "unknown", None),
    ("Tolong tuliskan kode Python untuk mengurutkan list", "unknown", None),
]


async def main() -> None:
    rows = []
    for query, expected_intent, expected_ticket_id in LABELED_QUERIES:
        intent, method = await router.detect_intent(query)
        keyword = router.keyword_intent(query)
        ticket_id = router.extract_ticket_id(query) if intent == "ticket_status" else None
        rows.append({
            "query": query,
            "expected_intent": expected_intent,
            "keyword_baseline": keyword,
            "keyword_correct": keyword == expected_intent,
            "router_intent": intent,
            "router_method": method,
            "router_correct": intent == expected_intent,
            "expected_ticket_id": expected_ticket_id or "",
            "extracted_ticket_id": ticket_id or "",
            "ticket_id_correct": ticket_id == expected_ticket_id if expected_intent == "ticket_status" else "",
        })

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(df[["query", "expected_intent", "keyword_baseline", "router_intent", "router_method"]].to_string(index=False))
    print(f"\nAkurasi keyword baseline : {df['keyword_correct'].sum()}/{len(df)}")
    print(f"Akurasi router final     : {df['router_correct'].sum()}/{len(df)}")
    tickets = df[df["expected_intent"] == "ticket_status"]
    print(f"ticket_id tepat          : {(tickets['ticket_id_correct'] == True).sum()}/{len(tickets)}")  # noqa: E712
    print(f"Hasil lengkap            : {OUTPUT_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    asyncio.run(main())
