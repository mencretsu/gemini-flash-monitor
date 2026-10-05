#!/usr/bin/env python3
"""Daily Gemini Flash-family deprecation monitor.

Sources:
1) Gemini Developer API models.list -> discovers all currently exposed models.
2) Official Gemini deprecations page -> deprecation/shutdown schedule.

The script is intentionally fail-closed: if the official deprecations page or
Gemini API cannot be read, it exits non-zero and does not overwrite state.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

GEMINI_MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models"
DEPRECATIONS_URL = "https://ai.google.dev/gemini-api/docs/deprecations"
TELEGRAM_URL = "https://api.telegram.org/bot{token}/sendMessage"
STATE_PATH = Path("state.json")
TIMEOUT = 30
PAGE_SIZE = 1000


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "GeminiFlashMonitor/1.0 (+GitHub Actions)",
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        }
    )
    return session


def fetch_all_gemini_models(session: requests.Session, api_key: str) -> list[dict[str, Any]]:
    """Fetch every page from models.list."""
    all_models: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        params: dict[str, Any] = {"key": api_key, "pageSize": PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token

        response = session.get(GEMINI_MODELS_URL, params=params, timeout=TIMEOUT)
        if response.status_code != 200:
            body = response.text[:1000].replace("\n", " ")
            raise RuntimeError(
                f"Gemini models.list failed: HTTP {response.status_code}: {body}"
            )

        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            raise RuntimeError("Gemini models.list returned an unexpected JSON shape")

        all_models.extend(m for m in models if isinstance(m, dict))
        page_token = payload.get("nextPageToken")
        if not page_token:
            break

    return all_models


def model_id_from_api_model(model: dict[str, Any]) -> str | None:
    # Prefer the exact resource name because deprecation schedules can target
    # a versioned endpoint (for example, a model ending in -preview or -001).
    name = model.get("name")
    if isinstance(name, str) and name.startswith("models/"):
        candidate = name[len("models/") :].strip()
        if candidate:
            return candidate

    base = model.get("baseModelId")
    if isinstance(base, str) and base.strip():
        return base.strip()
    return None


def discover_flash_models(api_models: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return all API-discoverable models whose ID or name contains 'flash'."""
    discovered: dict[str, dict[str, Any]] = {}

    for model in api_models:
        model_id = model_id_from_api_model(model)
        if not model_id:
            continue

        name = str(model.get("name", ""))
        haystack = f"{model_id} {name}".lower()
        if "flash" not in haystack:
            continue

        methods = model.get("supportedGenerationMethods")
        if not isinstance(methods, list):
            methods = []

        discovered[model_id] = {
            "model": model_id,
            "display_name": str(model.get("displayName", "")).strip(),
            "version": str(model.get("version", "")).strip(),
            "supported_methods": sorted(
                str(x) for x in methods if isinstance(x, (str, int, float))
            ),
            "source": "models.list",
        }

    return discovered


def normalize_date_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def parse_date(value: str) -> str | None:
    text = normalize_date_text(value)
    if not text or "no shutdown date" in text.lower():
        return None

    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass

    # Be tolerant of dates embedded in extra text.
    match = re.search(r"\b([A-Z][a-z]+\s+\d{1,2},\s+\d{4})\b", text)
    if match:
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(match.group(1), fmt).date().isoformat()
            except ValueError:
                pass

    raise ValueError(f"Unrecognized shutdown date text: {value!r}")


def clean_cell_text(cell: Any) -> str:
    text = cell.get_text(" ", strip=True)
    text = normalize_date_text(text)
    # Markdown/HTML code formatting sometimes leaves surrounding punctuation.
    text = text.strip("` ")
    return text


def extract_model_code(cell: Any) -> str | None:
    code = cell.find("code")
    candidate = code.get_text(" ", strip=True) if code else clean_cell_text(cell)
    candidate = candidate.strip("` ").strip()
    if not candidate:
        return None

    # Model IDs are URL-safe-ish lowercase identifiers used by Gemini docs.
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", candidate):
        return candidate
    return None


def fetch_deprecation_schedule(session: requests.Session) -> dict[str, dict[str, Any]]:
    """Parse the official deprecation tables into model -> schedule records."""
    response = session.get(DEPRECATIONS_URL, timeout=TIMEOUT)
    if response.status_code != 200:
        body = response.text[:1000].replace("\n", " ")
        raise RuntimeError(
            f"Official deprecations page failed: HTTP {response.status_code}: {body}"
        )

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        raise RuntimeError("Official deprecations page contained no HTML tables")

    schedule: dict[str, dict[str, Any]] = {}
    relevant_rows = 0

    for table in tables:
        rows = table.find_all("tr")
        if not rows:
            continue

        headers = [clean_cell_text(c).lower() for c in rows[0].find_all(["th", "td"])]
        # We only want model tables with a shutdown-date column.
        if not any("model" == h or h.startswith("model ") for h in headers):
            continue
        if not any("shutdown" in h for h in headers):
            continue

        model_idx = next(
            (i for i, h in enumerate(headers) if h == "model" or h.startswith("model ")),
            0,
        )
        shutdown_idx = next(i for i, h in enumerate(headers) if "shutdown" in h)
        replacement_idx = next(
            (i for i, h in enumerate(headers) if "replacement" in h), None
        )

        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if len(cells) <= max(model_idx, shutdown_idx):
                continue

            model_id = extract_model_code(cells[model_idx])
            if not model_id or "flash" not in model_id.lower():
                continue

            shutdown_raw = clean_cell_text(cells[shutdown_idx])
            shutdown = parse_date(shutdown_raw)

            replacement = None
            if replacement_idx is not None and replacement_idx < len(cells):
                replacement = extract_model_code(cells[replacement_idx])

            schedule[model_id] = {
                "shutdown": shutdown,
                "replacement": replacement,
                "source": "official_deprecations",
            }
            relevant_rows += 1

    if relevant_rows == 0:
        raise RuntimeError(
            "Official deprecations page was fetched, but no Flash model rows were parsed"
        )

    return schedule


def status_for(model_id: str, shutdown: str | None, today: date) -> str:
    if shutdown is None:
        return "active"
    shutdown_date = date.fromisoformat(shutdown)
    return "shutdown" if shutdown_date <= today else "deprecated"


def build_snapshot(
    api_flash: dict[str, dict[str, Any]],
    deprecations: dict[str, dict[str, Any]],
    today: date,
) -> dict[str, dict[str, Any]]:
    """Merge API discovery with official deprecation data."""
    model_ids = set(api_flash) | set(deprecations)
    snapshot: dict[str, dict[str, Any]] = {}

    for model_id in sorted(model_ids):
        api_info = api_flash.get(model_id, {})
        dep_info = deprecations.get(model_id, {})
        shutdown = dep_info.get("shutdown")
        replacement = dep_info.get("replacement")

        snapshot[model_id] = {
            "status": status_for(model_id, shutdown, today),
            "shutdown": shutdown,
            "replacement": replacement,
            "display_name": api_info.get("display_name", ""),
            "version": api_info.get("version", ""),
            "supported_methods": api_info.get("supported_methods", []),
            "in_api": model_id in api_flash,
        }

    return snapshot


def load_previous_state() -> dict[str, dict[str, Any]] | None:
    if not STATE_PATH.exists():
        return None

    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not read {STATE_PATH}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        raise RuntimeError(f"Invalid state format in {STATE_PATH}")

    return data["models"]


def save_state(snapshot: dict[str, dict[str, Any]]) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "models": snapshot,
    }
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(STATE_PATH)


def short_model_name(model: str) -> str:
    return model.replace("gemini-", "", 1)


def days_until(shutdown: str | None, today: date) -> int | None:
    if not shutdown:
        return None
    return (date.fromisoformat(shutdown) - today).days


def build_message(
    snapshot: dict[str, dict[str, Any]],
    previous: dict[str, dict[str, Any]] | None,
    today: date,
) -> str:
    active = [m for m, v in snapshot.items() if v["status"] == "active"]
    deprecated = [m for m, v in snapshot.items() if v["status"] == "deprecated"]
    shutdown = [m for m, v in snapshot.items() if v["status"] == "shutdown"]

    lines = [
        "🤖 <b>Gemini Flash Monitor</b>",
        f"📅 {today.isoformat()}",
        "",
        f"🟢 Active: <b>{len(active)}</b>",
        f"⚠️ Deprecated: <b>{len(deprecated)}</b>",
        f"🔴 Shutdown: <b>{len(shutdown)}</b>",
    ]

    if previous is None:
        lines += [
            "",
            "<b>Initial baseline saved.</b>",
        ]
    else:
        changes: list[str] = []
        all_ids = sorted(set(previous) | set(snapshot))

        for model in all_ids:
            old = previous.get(model)
            new = snapshot.get(model)

            if old is None and new is not None:
                if new["status"] in {"deprecated", "shutdown"}:
                    changes.append(
                        f"🆕 <code>{model}</code> → <b>{new['status'].upper()}</b>"
                    )
                elif new.get("in_api"):
                    changes.append(f"🆕 <code>{model}</code> → <b>NEW MODEL</b>")
                continue

            if old is not None and new is None:
                # Do not call a temporary API disappearance a shutdown.
                continue

            assert old is not None and new is not None
            if old.get("status") != new.get("status"):
                changes.append(
                    f"🔄 <code>{model}</code>: "
                    f"{str(old.get('status', '?')).upper()} → "
                    f"<b>{str(new.get('status', '?')).upper()}</b>"
                )

            if old.get("shutdown") != new.get("shutdown") and new.get("shutdown"):
                changes.append(
                    f"📆 <code>{model}</code>: shutdown → "
                    f"<b>{new['shutdown']}</b>"
                )

            if old.get("replacement") != new.get("replacement") and new.get("replacement"):
                changes.append(
                    f"➡️ <code>{model}</code>: replacement → "
                    f"<code>{new['replacement']}</code>"
                )

        if changes:
            lines += ["", "<b>Changes:</b>"]
            lines.extend(changes[:30])
            if len(changes) > 30:
                lines.append(f"… and {len(changes) - 30} more")
        else:
            lines += ["", "✅ No lifecycle changes since the previous run."]

    future_deadlines = []
    for model in deprecated:
        left = days_until(snapshot[model].get("shutdown"), today)
        if left is not None:
            future_deadlines.append((date.fromisoformat(snapshot[model]["shutdown"]), model, left))

    if future_deadlines:
        future_deadlines.sort()
        lines += ["", "<b>Upcoming shutdowns:</b>"]
        for shutdown_date, model, left in future_deadlines[:15]:
            replacement = snapshot[model].get("replacement")
            replacement_text = f" → <code>{replacement}</code>" if replacement else ""
            lines.append(
                f"• <code>{model}</code> — {shutdown_date.isoformat()} "
                f"({left}d){replacement_text}"
            )
        if len(future_deadlines) > 15:
            lines.append(f"• … {len(future_deadlines) - 15} more")

    lines.append("")
    lines.append(
        'Source: <a href="https://ai.google.dev/gemini-api/docs/deprecations">Google Gemini deprecations</a>'
    )
    return "\n".join(lines)


def send_telegram(session: requests.Session, token: str, chat_id: str, message: str) -> None:
    url = TELEGRAM_URL.format(token=token)
    response = session.post(
        url,
        data={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        },
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        body = response.text[:1000].replace("\n", " ")
        raise RuntimeError(f"Telegram send failed: HTTP {response.status_code}: {body}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Telegram returned invalid JSON") from exc

    if not payload.get("ok"):
        raise RuntimeError(f"Telegram API returned ok=false: {payload}")


def main() -> int:
    api_key = require_env("GEMINI_API_KEY")
    bot_token = require_env("TELEGRAM_BOT_TOKEN")
    chat_id = require_env("TELEGRAM_CHAT_ID")

    today = datetime.now(timezone.utc).date()
    session = get_session()

    print("[1/4] Fetching Gemini models...")
    api_models = fetch_all_gemini_models(session, api_key)
    api_flash = discover_flash_models(api_models)
    print(f"      Found {len(api_flash)} Flash-family models via models.list")

    print("[2/4] Fetching official deprecation schedule...")
    deprecations = fetch_deprecation_schedule(session)
    print(f"      Parsed {len(deprecations)} Flash-family deprecation rows")

    print("[3/4] Building snapshot...")
    snapshot = build_snapshot(api_flash, deprecations, today)
    previous = load_previous_state()
    message = build_message(snapshot, previous, today)

    # Always send one daily report. This matches the requested daily cadence.
    print("[4/4] Sending Telegram report...")
    send_telegram(session, bot_token, chat_id, message)

    save_state(snapshot)
    print(f"Saved state for {len(snapshot)} models to {STATE_PATH}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
