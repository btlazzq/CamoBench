import asyncio
import random
from typing import Optional

import httpx

from .config import DEFAULT_HEADERS, NET_SEM, RETRY_BASE_SLEEP, RETRY_TIMES, TIMEOUT



def get_proxy_url() -> Optional[str]:
    import os
    return os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or os.getenv("ALL_PROXY")



def build_async_client() -> httpx.AsyncClient:
    import os

    proxy = get_proxy_url()
    transport = httpx.AsyncHTTPTransport(
        proxy=proxy,
        retries=0,
        limits=httpx.Limits(
            max_connections=int(os.getenv("MAX_CONNECTIONS", "20")),
            max_keepalive_connections=int(os.getenv("MAX_KEEPALIVE", "10")),
            keepalive_expiry=float(os.getenv("KEEPALIVE_EXPIRY", "20")),
        ),
    )
    use_http2 = os.getenv("HTTP2", "0") == "1"
    return httpx.AsyncClient(
        timeout=TIMEOUT,
        follow_redirects=True,
        transport=transport,
        headers=DEFAULT_HEADERS,
        http2=use_http2,
    )


async def fetch_json(client: httpx.AsyncClient, url: str, params=None, headers=None) -> dict:
    last_exc = None
    req_headers = dict(DEFAULT_HEADERS)
    if headers:
        req_headers.update(headers)

    for attempt in range(RETRY_TIMES):
        try:
            async with NET_SEM:
                r = await client.get(url, params=params, headers=req_headers)
            r.raise_for_status()
            return r.json()
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError,
                httpx.ConnectError, httpx.ReadError) as e:
            last_exc = e
        except httpx.HTTPStatusError as e:
            last_exc = e
            status = e.response.status_code
            if status not in (429, 500, 502, 503, 504):
                raise
        sleep = RETRY_BASE_SLEEP * (2 ** attempt) + random.uniform(0, 0.3)
        await asyncio.sleep(sleep)
    raise last_exc
