"""
Modul Tools — fungsi-fungsi "aksi" yang dipanggil router ketika intent
BUKAN pertanyaan informasi/kebijakan, melainkan permintaan spesifik
yang butuh pengecekan data langsung (bukan retrieval semantik).

Pola ini mengikuti konsep Tool Integration di Sesi 12.3.
"""

from functools import cache
from pathlib import Path

import pandas as pd

MOCK_TICKETS_PATH = Path(__file__).resolve().parent.parent / "data" / "mock_tickets.csv"

# employee_name sengaja tidak dikembalikan: API belum memverifikasi identitas peminta,
# sehingga siapa pun yang tahu nomor tiket tidak boleh melihat nama pemilik tiket.
PUBLIC_FIELDS = ["ticket_id", "category", "subject", "status", "last_update"]


@cache
def load_tickets() -> dict[str, dict]:
    """Muat mock_tickets.csv sekali, diindeks berdasarkan ticket_id (huruf besar) untuk lookup O(1)."""
    df = pd.read_csv(MOCK_TICKETS_PATH, dtype=str)
    return {row["ticket_id"].strip().upper(): row for row in df[PUBLIC_FIELDS].to_dict("records")}


def find_ticket(ticket_id: str) -> dict | None:
    """Kembalikan data tiket, atau None jika ticket_id tidak ada di data."""
    return load_tickets().get(ticket_id.strip().upper())


def get_ticket_status(ticket_id: str) -> str:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.
    Jika ticket_id tidak ditemukan, mengembalikan pesan informatif (bukan exception).
    """
    ticket = find_ticket(ticket_id)
    if ticket is None:
        return (
            f"Tiket {ticket_id} tidak ditemukan. Pastikan nomor tiket sudah benar (format TICKET-XXXX), "
            "atau hubungi tim IT Support jika tiket baru saja dibuat."
        )
    return (
        f"Tiket {ticket['ticket_id']} ({ticket['subject']}, kategori {ticket['category']}) "
        f"saat ini berstatus: {ticket['status']}. Pembaruan terakhir: {ticket['last_update']}."
    )
