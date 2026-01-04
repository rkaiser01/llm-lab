from datetime import datetime

def ts() -> str:
    """
    Return a formatted timestamp for console output.
    """
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
