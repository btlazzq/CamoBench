from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from .config import HTTP_TIMEOUT, VERIFY_SSL, BAIDU_USER_AGENT

def build_async_client() -> httpx.AsyncClient:

    headers = {

        'User-Agent': BAIDU_USER_AGENT,

        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',

    }

    return httpx.AsyncClient(timeout=HTTP_TIMEOUT, verify=VERIFY_SSL, follow_redirects=True, headers=headers)

async def fetch_json(

    client: httpx.AsyncClient,

    url: str,

    *,

    params: Optional[Dict[str, Any]] = None,

    headers: Optional[Dict[str, str]] = None,

    method: str = 'GET',

    json_body: Optional[Dict[str, Any]] = None,

    timeout: Optional[float] = None,

) -> Dict[str, Any]:

    response = await client.request(

        method, url, params=params, headers=headers, json=json_body, timeout=timeout

    )

    response.raise_for_status()

    data = response.json()

    return data if isinstance(data, dict) else {'result': data}

async def fetch_text(

    client: httpx.AsyncClient,

    url: str,

    *,

    params: Optional[Dict[str, Any]] = None,

    headers: Optional[Dict[str, str]] = None,

) -> str:

    response = await client.get(url, params=params, headers=headers)

    response.raise_for_status()

    return response.text
