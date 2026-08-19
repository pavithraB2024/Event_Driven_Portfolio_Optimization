"""Async remote logger via ntfy.sh for distributed training monitoring.

Disabled by default. Set ENABLE_REMOTE_LOGGING=1 to stream logs externally.
"""
import os
import queue
import threading


class RemoteLogger:
    def __init__(self, topic: str):
        self._enabled = os.getenv("ENABLE_REMOTE_LOGGING", "0") == "1"
        if self._enabled:
            import requests  # noqa: F401
            self._requests = requests
            self.url = f"https://ntfy.sh/{topic}"
            self.queue: queue.Queue[str] = queue.Queue()
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def log(self, message: str):
        print(message, flush=True)
        if self._enabled:
            self.queue.put(message)

    def _worker(self):
        while True:
            message = self.queue.get()
            try:
                self._requests.post(self.url, data=message.encode('utf-8'), timeout=3)
            except Exception:
                pass
            self.queue.task_done()
