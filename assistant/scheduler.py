"""APScheduler wrapper around the monitor tick.

Disabled by default (config.monitor.enabled=False). When enabled, calling
`run_forever()` blocks and ticks every `config.monitor.interval_minutes`.
"""

from __future__ import annotations

import logging
import signal
import time
import uuid

from apscheduler.schedulers.background import BackgroundScheduler

from assistant.config import get_config
from assistant.graph import get_graph

log = logging.getLogger(__name__)


class MonitorScheduler:
    def __init__(self) -> None:
        self._sched = BackgroundScheduler()
        self._configured = False

    def configure(self) -> bool:
        cfg = get_config().monitor
        if not cfg.enabled:
            log.info("Monitor is disabled in config (monitor.enabled=false).")
            return False
        graph = get_graph()
        self._sched.add_job(
            lambda: graph.invoke(
                {"intent": "monitor_tick"},
                config={"configurable": {"thread_id": str(uuid.uuid4())}},
            ),
            trigger="interval",
            minutes=cfg.interval_minutes,
            id="monitor_tick",
            replace_existing=True,
            next_run_time=None,
        )
        self._configured = True
        return True

    def run_forever(self) -> None:
        if not self._configured and not self.configure():
            return
        self._sched.start()
        log.info(
            "Monitor running every %d minutes. Ctrl-C to stop.",
            get_config().monitor.interval_minutes,
        )
        stop = False

        def _handle(_sig, _frame):
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, _handle)
        signal.signal(signal.SIGTERM, _handle)
        try:
            while not stop:
                time.sleep(1)
        finally:
            self._sched.shutdown(wait=False)
