"""JSONL and CSV progress-log I/O helpers shared by all pipeline stages."""

import os
import csv
import json
import time


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load all records from a JSONL file, skipping malformed/empty lines."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_single_jsonl_record(path):
    """Load the first (and typically only) record from a per-video/per-clip JSONL."""
    records = load_jsonl(path)
    return records[0] if records else None


def write_jsonl(path, records, append=False):
    """Write records to a JSONL file (one dict per line, UTF-8)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = 'a' if append else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_single_jsonl_record(path, record):
    """Write a single record to a JSONL file (overwrites)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl_record(path, record):
    """Append a single record to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CSV progress logs
# ---------------------------------------------------------------------------

class ProgressLog:
    """
    Minimal append-only CSV log used for resumability across all stages.
    Headers are written on first use; `read_processed(statuses)` returns
    the set of ids whose last logged status is in `statuses`.
    """

    def __init__(self, path, headers):
        self.path = path
        self.headers = list(headers)
        self._ensure_headers()

    def _ensure_headers(self):
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(self.headers)

    def append(self, **fields):
        row = [fields.get(h, "") for h in self.headers]
        with open(self.path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(row)

    def read_processed(self, statuses):
        """Return ids (first column) whose last occurrence has a status in `statuses`."""
        if not os.path.exists(self.path):
            return set()
        last_status = {}
        with open(self.path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_status[row[self.headers[0]]] = row.get("status", "")
        return {i for i, s in last_status.items() if s in statuses}


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")
