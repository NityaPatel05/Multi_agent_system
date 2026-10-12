"""SSRF-safe HTTP fetch for untrusted URLs (search-result URLs, or URLs sent to the
extract server).

Defences, in order: http(s) only; the hostname is resolved HERE and every resulting IP must
be public (no loopback / RFC1918 / link-local incl. cloud metadata 169.254.169.254 /
reserved / multicast / unspecified); the connection is pinned to the validated IP (so a DNS
rebind between check and connect can't swap it) with the original Host header and TLS
SNI/cert name; redirects are followed manually (max 3) and EVERY hop is re-validated;
proxy env vars are ignored; the body is streamed with a byte cap and a total wall-clock
deadline (not just a per-socket timeout).

`EXTRACT_ALLOW_PRIVATE=1` disables the IP check (local development/tests only).
"""

from __future__ import annotations

import ipaddress
import os
import socket
import time
from typing import List
from urllib.parse import urljoin, urlparse, urlunparse

MAX_REDIRECTS = 3
MAX_BYTES = int(os.environ.get("EXTRACT_MAX_BYTES", str(10 * 1024 * 1024)))
TOTAL_TIMEOUT_S = float(os.environ.get("EXTRACT_TOTAL_TIMEOUT_S", "30"))
_ALLOWED_SCHEMES = {"http", "https"}
# Many sites (e.g. Wikipedia) block anonymous bot UAs; set EXTRACT_USER_AGENT to something
# descriptive with a contact URL/email.
_UA = os.environ.get("EXTRACT_USER_AGENT", "research-langgraph/1.0")


class UnsafeURLError(ValueError):
    pass


def _allow_private() -> bool:
    return os.environ.get("EXTRACT_ALLOW_PRIVATE", "").strip().lower() in {"1", "true", "yes"}


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved
                or addr.is_multicast or addr.is_unspecified or not addr.is_global)


def resolve_public_ip(host: str, port: int) -> str:
    """Resolve `host` and return one IP, raising UnsafeURLError if ANY answer is non-public."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise UnsafeURLError(f"cannot resolve host {host!r}: {e}") from e
    ips: List[str] = list(dict.fromkeys(i[4][0] for i in infos))
    if not ips:
        raise UnsafeURLError(f"no addresses for {host!r}")
    if not _allow_private():
        bad = [ip for ip in ips if not _is_public(ip)]
        if bad:
            raise UnsafeURLError(f"refusing to fetch non-public address for host {host!r}")
    return ips[0]


def _validate(url: str):
    p = urlparse(url)
    if p.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"refusing to fetch non-http(s) URL scheme: {p.scheme!r}")
    if not p.hostname:
        raise UnsafeURLError("URL has no host")
    if p.username or p.password:
        raise UnsafeURLError("URLs with embedded credentials are not allowed")
    port = p.port or (443 if p.scheme.lower() == "https" else 80)
    return p, port


def fetch(url: str) -> bytes:
    import httpx  # lazy import

    deadline = time.monotonic() + TOTAL_TIMEOUT_S
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        p, port = _validate(current)
        ip = resolve_public_ip(p.hostname, port)
        host_ip = f"[{ip}]" if ":" in ip else ip
        pinned = urlunparse(p._replace(netloc=f"{host_ip}:{port}"))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("fetch exceeded total time budget")
        headers = {"User-Agent": _UA, "Host": p.netloc.split("@")[-1]}
        ext = {"sni_hostname": p.hostname} if p.scheme.lower() == "https" else {}
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=min(10.0, remaining)) as c:
            with c.stream("GET", pinned, headers=headers, extensions=ext) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    loc = resp.headers.get("location")
                    if not loc:
                        raise UnsafeURLError("redirect without Location")
                    current = urljoin(current, loc)
                    continue
                if resp.status_code >= 400:  # don't leak the pinned IP URL in the message
                    raise RuntimeError(f"HTTP {resp.status_code} from {p.hostname}")
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf += chunk
                    if len(buf) >= MAX_BYTES:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError("fetch exceeded total time budget")
                return bytes(buf[:MAX_BYTES])
    raise UnsafeURLError(f"too many redirects (>{MAX_REDIRECTS})")
