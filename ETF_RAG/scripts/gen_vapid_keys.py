"""VAPID 키쌍 생성 — 웹 푸시 알림용 (Phase F 푸시).

실행: python scripts/gen_vapid_keys.py
출력된 PUBLIC/PRIVATE를 .env (및 Railway env)에 설정:
  VAPID_PUBLIC_KEY=...   (프론트 applicationServerKey로도 노출 — /push/vapid-public-key)
  VAPID_PRIVATE_KEY=...
  VAPID_SUBJECT=mailto:you@example.com

키는 1회만 생성해 고정한다. 키를 바꾸면 기존 구독은 전부 무효가 된다.
"""

import base64

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid01


def main():
    v = Vapid01()
    v.generate_keys()

    pub = v.public_key.public_bytes(
        serialization.Encoding.X962,
        serialization.PublicFormat.UncompressedPoint,
    )
    pub_b64 = base64.urlsafe_b64encode(pub).rstrip(b"=").decode()

    priv_raw = v.private_key.private_numbers().private_value.to_bytes(32, "big")
    priv_b64 = base64.urlsafe_b64encode(priv_raw).rstrip(b"=").decode()

    print("# .env / Railway env 에 추가:")
    print(f"VAPID_PUBLIC_KEY={pub_b64}")
    print(f"VAPID_PRIVATE_KEY={priv_b64}")
    print("VAPID_SUBJECT=mailto:you@example.com")


if __name__ == "__main__":
    main()
