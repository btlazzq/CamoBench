import os
import asyncio
import httpx

UA = os.getenv("BAIKE_USER_AGENT", "CamouflageResearchBot/1.0 (contact: you@example.com)")
DEFAULT_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
}

TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

MAX_NET_CONCURRENCY = int(os.getenv("MAX_NET_CONCURRENCY", "6"))
NET_SEM = asyncio.Semaphore(MAX_NET_CONCURRENCY)

RETRY_TIMES = int(os.getenv("NET_RETRY_TIMES", "4"))
RETRY_BASE_SLEEP = float(os.getenv("NET_RETRY_BASE_SLEEP", "0.6"))

BAIKE_API_URL = os.getenv("BAIKE_API_URL", "https://appbuilder.baidu.com/v2/baike/lemma/get_content")
BAIKE_API_KEY = os.getenv("APPBUILDER_API_KEY", "").strip()

TERM_CONCURRENCY = int(os.getenv("TERM_CONCURRENCY", "2"))
TERM_SEM = asyncio.Semaphore(TERM_CONCURRENCY)
