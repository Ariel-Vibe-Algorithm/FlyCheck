"""Download reproducible raw data snapshots into data/raw/.

Run:  python -m src.download_data

Each download is recorded in data/raw/manifest.json with its source URL,
retrieval time (UTC), size and SHA-256, so the dataset can be reproduced.
Raw files are git-ignored; this script is how to rebuild them.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
USER_AGENT = "FlyCheck-academic-project/0.1"

SOURCES = {
    # The Octopus (HaTamnun) archive of the IAA flight board. License: CC BY.
    "odata_flydata_log": {
        "url": "https://www.odata.org.il/dataset/98b4f544-a7c8-4113-87de-7da1861c3a74/"
               "resource/d5ee74ca-2f56-4734-933e-02f4934420c2/download/flydata_log.csv",
        "path": "odata/flydata_log.csv",
        "license": "CC BY (The Octopus / odata.org.il)",
    },
    # Current IAA flight board (about 24h back to 3-4 days ahead). A one-time snapshot.
    "govil_board_snapshot": {
        "url": "https://data.gov.il/api/3/action/datastore_search"
               "?resource_id=e83f763b-b7d7-479e-b172-ae981ddc6de5&limit=32000",
        "path": "govil/board_{stamp}.json",
        "license": "data.gov.il - Other (Open)",
    },
    # Airline-level aggregates (one row per airline). Free to use with a link back.
    "tlvflights_airline_stats": {
        "url": "https://www.tlvflights.com/en/flight-statistics/export.csv?range=all",
        "path": "tlvflights/airline_stats.csv",
        "license": "tlvflights.com - free to use with attribution link",
    },
    # Israel CBS CPI component 121390: travel abroad and local flights (monthly).
    "cbs_travel_price_index": {
        "url": "https://api.cbs.gov.il/index/data/price?id=121390&format=json&lang=en"
               "&startPeriod=01-2015&endPeriod=12-2030&PageSize=1000",
        "path": "cbs/price_index_121390.json",
        "license": "Israel Central Bureau of Statistics (public)",
    },
}


def _download(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = response.read()
    dest.write_bytes(payload)
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = RAW_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}

    for name, source in SOURCES.items():
        dest = RAW_DIR / source["path"].format(stamp=stamp)
        print(f"Downloading {name} ...", flush=True)
        try:
            info = _download(source["url"], dest)
        except Exception as exc:  # keep going; report which source failed
            print(f"  FAILED: {exc}")
            continue
        manifest[name] = {
            "url": source["url"],
            "file": dest.relative_to(RAW_DIR).as_posix(),
            "license": source["license"],
            "retrieved_utc": stamp,
            **info,
        }
        print(f"  saved {dest.relative_to(RAW_DIR)} ({info['bytes']:,} bytes)")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
