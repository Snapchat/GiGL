"""Per-connection TCP throughput statistics sampled from ``ss -tin`` on a background thread.

Used by the loader's sampling statistics to tell whether individual flows are capped (for example
by a per-flow rate limit or a small congestion window) while the host NIC as a whole is not
saturated. Hosts without the ``ss`` binary report ``flow_samples=0``.
"""

import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from gigl.common.logger import Logger
from gigl.distributed.utils.host_stats import _percentile

logger = Logger()

_DEFAULT_INTERVAL_S = 30.0
_SS_TIMEOUT_S = 5.0
_NON_DATA_PREFIXES = ("127.", "169.254.", "fe80:", "::ffff:127.", "::ffff:169.254.")
_ACTIVE_RATE_MBPS = 1.0
_RATE_UNITS_MBPS = {"bps": 1e-6, "Kbps": 1e-3, "Mbps": 1.0, "Gbps": 1e3}
_RATE = re.compile(r"^([\d.]+)([KMG]?bps)$")
_DURATION = re.compile(r"^([\d.]+)(us|ms|s)")
_DURATION_MS = {"us": 1e-3, "ms": 1.0, "s": 1e3}


@dataclass
class Flow:
    """One established TCP connection as reported by ``ss -tin``.

    Rates are in Mbps, durations in ms. Fields ``ss`` did not print are ``None``.
    """

    local: str
    peer: str
    rtt_ms: Optional[float] = None
    cwnd: Optional[int] = None
    delivery_rate_mbps: Optional[float] = None
    send_mbps: Optional[float] = None
    retrans_total: int = 0
    bytes_acked: Optional[int] = None
    bytes_sent: Optional[int] = None
    busy_ms: Optional[float] = None
    rwnd_limited_ms: Optional[float] = None
    sndbuf_limited_ms: Optional[float] = None


def _parse_rate_mbps(token: str) -> Optional[float]:
    match = _RATE.match(token)
    if match is None:
        return None
    return float(match.group(1)) * _RATE_UNITS_MBPS[match.group(2)]


def _parse_duration_ms(token: str) -> Optional[float]:
    match = _DURATION.match(token)
    if match is None:
        return None
    return float(match.group(1)) * _DURATION_MS[match.group(2)]


def _host(address: str) -> str:
    """Strips the port (and any ``[]`` / interface suffix) from an ``ss`` address:port token."""
    host = address.rsplit(":", 1)[0].strip("[]")
    return host.split("%", 1)[0]


def _parse_info(flow: Flow, info: str) -> None:
    tokens = info.split()
    for i, token in enumerate(tokens):
        if token in ("send", "delivery_rate") and i + 1 < len(tokens):
            rate = _parse_rate_mbps(tokens[i + 1])
            if token == "send":
                flow.send_mbps = rate
            else:
                flow.delivery_rate_mbps = rate
            continue
        key, _, value = token.partition(":")
        if not value:
            continue
        if key == "rtt":
            flow.rtt_ms = float(value.split("/")[0])
        elif key == "cwnd":
            flow.cwnd = int(value)
        elif key == "retrans":
            flow.retrans_total = int(value.split("/")[-1])
        elif key == "bytes_acked":
            flow.bytes_acked = int(value)
        elif key == "bytes_sent":
            flow.bytes_sent = int(value)
        elif key == "busy":
            flow.busy_ms = _parse_duration_ms(value)
        elif key == "rwnd_limited":
            flow.rwnd_limited_ms = _parse_duration_ms(value)
        elif key == "sndbuf_limited":
            flow.sndbuf_limited_ms = _parse_duration_ms(value)


def parse_ss_tin(text: str) -> list[Flow]:
    """Parses ``ss -tin`` output (with or without ``-H``) into one ``Flow`` per socket.

    Each socket is a ``Recv-Q Send-Q Local:Port Peer:Port`` line (optionally preceded by a state
    column) followed by an indented line of TCP info; both may also be joined on one line.
    """
    flows: list[Flow] = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("Recv-Q") or line.startswith("State"):
            continue
        if line[0] in " \t":
            if flows:
                _parse_info(flows[-1], line)
            continue
        tokens = line.split()
        if not tokens[0].isdigit():
            tokens = tokens[1:]
        if len(tokens) < 4:
            continue
        flows.append(Flow(local=tokens[2], peer=tokens[3]))
        if len(tokens) > 4:
            _parse_info(flows[-1], " ".join(tokens[4:]))
    return flows


def is_data_flow(flow: Flow) -> bool:
    """True unless either end is loopback or link-local (metadata server and the like)."""
    for address in (flow.local, flow.peer):
        host = _host(address)
        if host.startswith(_NON_DATA_PREFIXES) or host == "::1":
            return False
    return True


def _run_ss() -> Optional[str]:
    """Runs ``ss -tinH state established``, falling back to ``ss -tin`` when ``-H`` is missing."""
    for args in (
        ["ss", "-tinH", "state", "established"],
        ["ss", "-tin", "state", "established"],
    ):
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=_SS_TIMEOUT_S, check=False
        )
        if result.returncode == 0:
            return result.stdout
    return None


class FlowStatsSampler:
    """Samples per-flow TCP statistics every ``interval_s`` on a daemon thread.

    The ``ss`` subprocess runs on the sampler's own thread, never on the thread that calls the
    reporter. Run one per host.

    Args:
        interval_s: Seconds between samples; defaults to ``GIGL_SAMPLING_FLOW_STATS_INTERVAL_S``
            (30 if unset: one call with 8000 flows takes about 170 ms).
        ss_runner: Returns ``ss -tin`` text for established sockets, or ``None`` on failure.
    """

    def __init__(
        self,
        interval_s: Optional[float] = None,
        ss_runner: Callable[[], Optional[str]] = _run_ss,
    ) -> None:
        if interval_s is None:
            interval_s = float(
                os.environ.get(
                    "GIGL_SAMPLING_FLOW_STATS_INTERVAL_S", _DEFAULT_INTERVAL_S
                )
            )
        self._interval_s = interval_s
        self._ss_runner = ss_runner
        self._enabled = shutil.which("ss") is not None or ss_runner is not _run_ss
        if not self._enabled:
            logger.warning("`ss` not found; per-flow TCP statistics are disabled")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reset_window()

    def _reset_window(self) -> None:
        self._samples = 0
        self._flows_total: list[int] = []
        self._flows_active: list[int] = []
        self._rates_mbps: list[float] = []
        self._rtts_ms: list[float] = []
        self._active_seen = 0
        self._rwnd_limited = 0
        self._sndbuf_limited = 0
        self._has_rwnd_field = False
        self._has_sndbuf_field = False
        self._retrans_total = 0

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(
            target=self._run, name="flow-stats-sampler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while True:
            try:
                self.sample()
            except Exception:
                logger.exception("Flow stats sample failed")
            if self._stop.wait(self._interval_s):
                return

    def sample(self) -> None:
        """Takes one ``ss`` snapshot and folds it into the current window."""
        text = self._ss_runner()
        if text is None:
            return
        flows = [flow for flow in parse_ss_tin(text) if is_data_flow(flow)]
        active = [
            flow
            for flow in flows
            if flow.delivery_rate_mbps is not None
            and flow.delivery_rate_mbps > _ACTIVE_RATE_MBPS
        ]
        with self._lock:
            self._samples += 1
            self._flows_total.append(len(flows))
            self._flows_active.append(len(active))
            self._rates_mbps.extend(
                flow.delivery_rate_mbps for flow in active if flow.delivery_rate_mbps
            )
            self._rtts_ms.extend(
                flow.rtt_ms for flow in flows if flow.rtt_ms is not None
            )
            for flow in active:
                self._active_seen += 1
                if flow.rwnd_limited_ms is not None:
                    self._has_rwnd_field = True
                    self._rwnd_limited += flow.rwnd_limited_ms > 0
                if flow.sndbuf_limited_ms is not None:
                    self._has_sndbuf_field = True
                    self._sndbuf_limited += flow.sndbuf_limited_ms > 0
            self._retrans_total = sum(flow.retrans_total for flow in flows)

    def make_reporter(self) -> Callable[[], dict[str, float]]:
        """Returns a ``SamplingStats`` reporter summarizing the samples since its last call."""
        return self._report

    def _report(self) -> dict[str, float]:
        with self._lock:
            report: dict[str, float] = {"flow_samples": float(self._samples)}
            if self._samples:
                report["flows_total"] = sum(self._flows_total) / self._samples
                report["flows_active"] = sum(self._flows_active) / self._samples
                report["flow_retrans_total"] = float(self._retrans_total)
            if self._rates_mbps:
                report["flow_delivery_rate_mbps_max"] = max(self._rates_mbps)
                report["flow_delivery_rate_mbps_p50"] = _percentile(
                    self._rates_mbps, 0.5
                )
                report["flow_delivery_rate_mbps_p99"] = _percentile(
                    self._rates_mbps, 0.99
                )
            if self._rtts_ms:
                report["flow_rtt_ms_p50"] = _percentile(self._rtts_ms, 0.5)
                report["flow_rtt_ms_p99"] = _percentile(self._rtts_ms, 0.99)
            if self._active_seen and self._has_rwnd_field:
                report["flows_rwnd_limited_frac"] = (
                    self._rwnd_limited / self._active_seen
                )
            if self._active_seen and self._has_sndbuf_field:
                report["flows_sndbuf_limited_frac"] = (
                    self._sndbuf_limited / self._active_seen
                )
            self._reset_window()
            return report
