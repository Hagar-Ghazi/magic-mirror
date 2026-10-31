"""Pick ICE servers for WebRTC so live video can connect from Streamlit Cloud.

A TURN relay is needed when the browser and server cannot reach each other directly.
Credentials come from env vars or Streamlit secrets, checked in this order:

1. TURN_URLS + TURN_USERNAME + TURN_CREDENTIAL   any static TURN server (e.g. Metered)
2. CLOUDFLARE_TURN_KEY_ID + CLOUDFLARE_TURN_KEY_API_TOKEN
3. TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN
4. HF_TOKEN                                      Hugging Face relay
"""

import os

import streamlit as st
from streamlit_webrtc.credentials import get_cloudflare_ice_servers, get_hf_ice_servers, get_twilio_ice_servers

STUN = {"urls": ["stun:stun.l.google.com:19302"]}
KEYS = [
    "TURN_URLS", "TURN_USERNAME", "TURN_CREDENTIAL",
    "CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN",
    "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN",
    "HF_TOKEN",
]


def _secret(key):
    value = os.getenv(key)
    if value:
        return value
    try:
        return str(st.secrets[key]) if key in st.secrets else None
    except Exception:  # no secrets file locally
        return None


def _static():
    urls, user, cred = _secret("TURN_URLS"), _secret("TURN_USERNAME"), _secret("TURN_CREDENTIAL")
    if not (urls and user and cred):
        return None
    return [{"urls": [u.strip() for u in urls.split(",") if u.strip()], "username": user, "credential": cred}]


def _urls(server):
    urls = server.get("urls") or server.get("url") if isinstance(server, dict) else getattr(server, "urls", None)
    return [urls] if isinstance(urls, str) else [str(u) for u in (urls or [])]


PROVIDERS = [
    ("Custom", ["TURN_URLS", "TURN_USERNAME", "TURN_CREDENTIAL"], lambda s: _static()),
    ("Cloudflare", ["CLOUDFLARE_TURN_KEY_ID", "CLOUDFLARE_TURN_KEY_API_TOKEN"],
     lambda s: get_cloudflare_ice_servers(s[0], s[1])),
    ("Twilio", ["TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN"], lambda s: get_twilio_ice_servers(s[0], s[1])),
    ("Hugging Face", ["HF_TOKEN"], lambda s: get_hf_ice_servers(s[0])),
]


def ice_servers():
    """Return (ice_servers, status_message, ok). Failures are reported, never hidden."""
    errors = []
    for name, keys, fetch in PROVIDERS:
        values = [_secret(k) for k in keys]
        if not all(values):
            if any(values):
                errors.append(f"{name}: missing {', '.join(k for k, v in zip(keys, values) if not v)}")
            continue
        try:
            servers = list(fetch(values))
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            continue
        if any(u.startswith(("turn:", "turns:")) for s in servers for u in _urls(s)):
            return servers + [STUN], f"Live video relay: {name} TURN", True
        errors.append(f"{name}: returned no TURN server")
    message = "No TURN relay, so live video may not connect on the cloud."
    if errors:
        message += " " + "; ".join(errors)
    return [STUN], message, False
