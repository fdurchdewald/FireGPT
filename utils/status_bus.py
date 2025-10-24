# status_bus.py
import threading

_lock   = threading.Lock()
_status = ""                 # aktueller Status-String

def set_status(msg: str) -> None:
    """Von langen Jobs aufrufen, um den Fortschritt zu setzen."""
    global _status
    with _lock:
        _status = msg

def get_status() -> str:
    """Vom Poll-Callback abfragen."""
    with _lock:
        return _status
