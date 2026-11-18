"""Web Push from the laptop to the owner's phone (VAPID keys and subscriptions stay in live/.keys)."""

import base64
import json
from pathlib import Path

KEYS = Path(__file__).resolve().parent / ".keys"


def vapid_public_key():
    # creates the key pair on first use; returns the URL-safe public key the site subscribes with
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    KEYS.mkdir(exist_ok=True)
    pem = KEYS / "vapid_private.pem"
    if not pem.exists():
        key = ec.generate_private_key(ec.SECP256R1())
        pem.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    key = serialization.load_pem_private_key(pem.read_bytes(), None)
    raw = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def parse_subscription(b64):
    sub = json.loads(base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode())
    if not sub.get("endpoint", "").startswith("https://") or "keys" not in sub:
        raise ValueError("not a push subscription")
    return sub


def save_subscription(sub):
    KEYS.mkdir(exist_ok=True)
    (KEYS / "subscription.json").write_text(json.dumps(sub))


def load_subscription():
    f = KEYS / "subscription.json"
    return json.loads(f.read_text()) if f.exists() else None


def alert_text(changer, race, later):
    if later:
        return "Something big happened", "Open when you're ready. No spoilers here."
    pct = lambda v: f"{round(v * 100)}%"
    return f"Race changer · {race} lap {changer['lap']}", f"{changer['reason']}. {changer['driver']} win chance {pct(changer['from'])} → {pct(changer['to'])}."


def send(title, body, url, sub=None):
    from pywebpush import WebPushException, webpush

    sub = sub or load_subscription()
    if not sub:
        print("push: no phone subscribed yet")
        return False
    try:
        webpush(sub, json.dumps({"title": title, "body": body, "url": url}), vapid_private_key=str(KEYS / "vapid_private.pem"),
                vapid_claims={"sub": "mailto:noreply@example.invalid"})
        return True
    except WebPushException as e:
        print("push failed:", e)
        return False
