"""Derive the URL-lexical features used by the Zieni phishing dataset."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse

import pandas as pd

_COUNTABLE = {
    "dots": ".", "hyph": "-", "underline": "_", "slash": "/", "questionmark": "?",
    "equal": "=", "at_sign": "@", "and": "&", "exclamation": "!", "space": " ", "tilde": "~",
    "comma": ",", "plus": "+", "asterisk": "*", "dollar": "$", "percent": "%",
}
_KNOWN_TLD = re.compile(r"\.(?:com|org|net|edu|gov|io|co|uk|de|fr|jp|au|us|info|biz|online|site|xyz)\b", re.I)
_EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)


def _count(text: str, name: str) -> int:
    return text.count(_COUNTABLE[name])


def _column_name(segment: str, symbol_name: str) -> str:
    """Match the supplied dataset's historical names for @ and ~ fields."""
    if symbol_name in {"at_sign", "tilde"}:
        return f"{symbol_name}_{segment}"
    return f"num_{symbol_name}_{segment}"


def _is_ip_address(value: str) -> int:
    try:
        ipaddress.ip_address(value.strip("[]"))
        return 1
    except ValueError:
        return 0


def _prepare_url(url: str) -> tuple[str, str, str, str, str]:
    raw = url.strip()
    if not raw:
        raise ValueError("Enter a URL to analyze.")
    parseable = raw if re.match(r"^[a-z][a-z0-9+.-]*://", raw, re.I) else f"http://{raw}"
    parsed = urlparse(parseable)
    domain = (parsed.hostname or "").lower()
    path = parsed.path or ""
    filename = path.rsplit("/", 1)[-1] if path and not path.endswith("/") else ""
    return raw, domain, path, filename, parsed.query or ""


def derive_url_features(url: str) -> dict[str, int]:
    """Return all 74 URL-lexical feature names in the supplied Zieni CSV."""
    raw, domain, path, filename, params = _prepare_url(url)
    values: dict[str, int] = {}

    for name in _COUNTABLE:
        values[_column_name("url", name)] = _count(raw, name)
    # The dataset calls this a singular feature; it is the number of # characters.
    values["hashtag_url"] = raw.count("#")
    values["num_tld_url"] = len(_KNOWN_TLD.findall(raw))
    values["length_url"] = len(raw)
    values["email_in_url"] = int(bool(_EMAIL.search(raw)))

    for name in ("dots", "hyph", "underline"):
        values[_column_name("dom", name)] = _count(domain, name)
    values["num_vowels_dom"] = sum(character in "aeiou" for character in domain)
    values["length_dom"] = len(domain)
    values["dom_in_ip"] = _is_ip_address(domain)
    values["server_client_dom"] = int("server" in domain or "client" in domain)

    for name in ("dots", "hyph", "underline", "slash", "equal", "at_sign", "and", "exclamation", "space", "tilde", "comma", "plus", "asterisk", "percent"):
        values[_column_name("path", name)] = _count(path, name)
    values["length_path"] = len(path)

    for name in ("dots", "hyph", "underline", "equal", "at_sign", "and", "exclamation", "space", "tilde", "comma", "plus", "percent"):
        values[_column_name("file", name)] = _count(filename, name)
    values["length_file"] = len(filename)

    for name in ("dots", "hyph", "underline", "slash", "questionmark", "equal", "at_sign", "and", "exclamation", "space", "tilde", "comma", "plus", "asterisk", "dollar", "percent"):
        values[_column_name("param", name)] = _count(params, name)
    values["length_param"] = len(params)
    values["tld_in_param"] = int(bool(_KNOWN_TLD.search(params)))
    values["num_param"] = 0 if not params else len([part for part in params.split("&") if part])
    return values


def url_to_model_frame(url: str, expected_features: list[str]) -> pd.DataFrame:
    """Build one row in precisely the training feature order, refusing unknown fields."""
    values = derive_url_features(url)
    unsupported = [feature for feature in expected_features if feature not in values]
    if unsupported:
        raise ValueError(
            "The saved model contains features this URL extractor does not support: "
            + ", ".join(unsupported)
        )
    return pd.DataFrame([{feature: values[feature] for feature in expected_features}], columns=expected_features)
