import urllib.request


def is_healthy(api_url: str) -> bool:
    try:
        urllib.request.urlopen(f"{api_url}/health", timeout=5)
        return True
    except OSError:
        return False
