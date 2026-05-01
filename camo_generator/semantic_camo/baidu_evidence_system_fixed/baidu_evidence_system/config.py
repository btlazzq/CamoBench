import os

APPBUILDER_API_KEY = os.getenv('APPBUILDER_API_KEY', '').strip()

AI_SEARCH_API_URL = os.getenv(

    'AI_SEARCH_API_URL',

    'https://qianfan.baidubce.com/v2/ai_search/chat/completions',

).strip()

BAIKE_API_KEY = APPBUILDER_API_KEY

BAIKE_API_URL = (os.getenv('BAIKE_API_URL') or os.getenv('AI_SEARCH_API_URL') or 'https://qianfan.baidubce.com/v2/ai_search/chat/completions').strip()

BAIDU_SEARCH_URL = os.getenv('BAIDU_SEARCH_URL', 'https://www.baidu.com/s').strip()

BAIDU_USER_AGENT = os.getenv(

    'BAIDU_USER_AGENT',

    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '

    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36',

).strip()

HTTP_TIMEOUT = float(os.getenv('HTTP_TIMEOUT', '20'))

AI_SEARCH_TIMEOUT = float(os.getenv('AI_SEARCH_TIMEOUT', '90'))

VERIFY_SSL = os.getenv('VERIFY_SSL', 'true').lower() not in {'0', 'false', 'no'}

MAX_TEXT_LEN = int(os.getenv('MAX_TEXT_LEN', '180'))
