"""Host-level network, memory and load counters, sampled from ``/proc`` on a background thread.

Used by the loader's sampling statistics to tell a network-bound or memory-bound host from a
CPU-bound one. Everything here reads Linux ``/proc`` files; on hosts without them the affected
fields are simply absent from the report.
"""

import math
import os
import threading
import time
from collections.abc import Sequence
from typing import Callable, Optional

from gigl.common.logger import Logger

logger = Logger()

_MB = 2**20
_GB = 2**30
_DEFAULT_NIC_LINE_RATE_GBPS = 50.0
_LINE_RATE_FRACTION = 0.8


def _percentile(values: Sequence[float], q: float) -> float:
    """Linearly interpolated ``q`` quantile (0..1) of a non-empty sequence."""
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _nic_line_rate_mb_s() -> float:
    gbps = float(
        os.environ.get("GIGL_SAMPLING_NIC_LINE_RATE_GBPS", _DEFAULT_NIC_LINE_RATE_GBPS)
    )
    return gbps * 1e9 / 8 / _MB


def _read(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def parse_net_dev(text: str) -> tuple[int, int, int, int]:
    """Sums ``/proc/net/dev`` over all interfaces except ``lo``.

    Returns:
        ``(rx_bytes, tx_bytes, rx_packets, tx_packets)``.
    """
    rx_bytes = tx_bytes = rx_pkts = tx_pkts = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, counters = line.split(":", 1)
        if name.strip() == "lo":
            continue
        fields = counters.split()
        rx_bytes += int(fields[0])
        rx_pkts += int(fields[1])
        tx_bytes += int(fields[8])
        tx_pkts += int(fields[9])
    return rx_bytes, tx_bytes, rx_pkts, tx_pkts


def _net_dev_interfaces(text: str) -> list[str]:
    names = [line.split(":", 1)[0].strip() for line in text.splitlines() if ":" in line]
    return [name for name in names if name != "lo"]


def _parse_header_value_pairs(text: str, section: str) -> dict[str, int]:
    """Parses a ``/proc/net/{snmp,netstat}`` section, which is a header line then a value line."""
    lines = [line for line in text.splitlines() if line.startswith(f"{section}:")]
    if len(lines) < 2:
        return {}
    names = lines[0].split()[1:]
    values = lines[1].split()[1:]
    return {name: int(value) for name, value in zip(names, values)}


def parse_snmp_tcp(text: str) -> dict[str, int]:
    """Parses the ``Tcp:`` section of ``/proc/net/snmp`` into ``{counter: value}``."""
    return _parse_header_value_pairs(text, "Tcp")


def parse_netstat_tcpext(text: str) -> dict[str, int]:
    """Parses the ``TcpExt:`` section of ``/proc/net/netstat`` into ``{counter: value}``."""
    return _parse_header_value_pairs(text, "TcpExt")


def _mem_available_gb() -> Optional[float]:
    text = _read("/proc/meminfo")
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / _GB
    return None


def _rss_gb(pids: Sequence[int]) -> float:
    total_kb = 0
    for pid in pids:
        text = _read(f"/proc/{pid}/status")
        if text is None:
            continue
        for line in text.splitlines():
            if line.startswith("VmRSS:"):
                total_kb += int(line.split()[1])
    return total_kb * 1024 / _GB


def _tcp_counters() -> dict[str, int]:
    counters: dict[str, int] = {}
    snmp = _read("/proc/net/snmp")
    if snmp is not None:
        counters["retrans_segs"] = parse_snmp_tcp(snmp).get("RetransSegs", 0)
    netstat = _read("/proc/net/netstat")
    if netstat is not None:
        tcpext = parse_netstat_tcpext(netstat)
        counters["timeouts"] = tcpext.get("TCPTimeouts", 0)
        counters["loss_probes"] = tcpext.get("TCPLossProbes", 0)
    return counters


class HostCounterSampler:
    """Samples host network, load and memory counters every ``interval_s`` on a daemon thread.

    The reporter from ``make_reporter`` summarizes the samples taken since its previous call
    (or since ``start``) and then starts a new window. It is safe to call from
    ``SamplingStats``'s window close because it only reads sampler state.

    Args:
        interval_s: Seconds between samples.
        pids: Processes whose resident memory is summed into ``rss_local_workers_gb_max``. The
            loader passes only the sampler workers of the rank that runs the host sampler, so the
            value excludes other ranks' workers on the same machine.
        net_interval_s: Seconds between samples of the fast network sampler, which reads only
            ``/proc/net/dev`` (via ``net_dev_reader``, ``clock`` for tests) so that sub-second bursts show up in ``*_100ms`` keys. The line
            rate used for ``*_bins_over_80pct_line`` comes from the
            ``GIGL_SAMPLING_NIC_LINE_RATE_GBPS`` env var (default 50).
    """

    def __init__(
        self,
        interval_s: float = 2.0,
        pids: Sequence[int] = (),
        net_interval_s: float = 0.1,
        net_dev_reader: Callable[[], Optional[str]] = lambda: _read("/proc/net/dev"),
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._interval_s = interval_s
        self._net_interval_s = net_interval_s
        self._net_dev_reader = net_dev_reader
        self._clock = clock
        self._line_rate_mb_s = _nic_line_rate_mb_s()
        self._fast_thread: Optional[threading.Thread] = None
        self._fast_last: Optional[tuple[int, int]] = None
        self._fast_last_time = 0.0
        self._pids = list(pids)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_net: Optional[tuple[int, int, int, int]] = None
        self._last_time = 0.0
        self._tcp_baseline: dict[str, int] = {}
        self._reset_window()

    def _reset_window(self) -> None:
        self._rx_mb_s: list[float] = []
        self._tx_mb_s: list[float] = []
        self._rx_pkt_s: list[float] = []
        self._tx_pkt_s: list[float] = []
        self._fast_rx_mb_s: list[float] = []
        self._fast_tx_mb_s: list[float] = []
        self._loadavg: list[float] = []
        self._mem_available_gb: list[float] = []
        self._rss_gb: list[float] = []
        self._samples = 0

    def start(self) -> None:
        net = self._net_dev_reader()
        if net is not None:
            logger.info(
                f"Host counter sampler watching interfaces {_net_dev_interfaces(net)}"
            )
            self._last_net = parse_net_dev(net)
            self._fast_last = self._last_net[:2]
            self._fast_last_time = self._clock()
            self._fast_thread = threading.Thread(
                target=self._run_fast_net, name="host-net-burst-sampler", daemon=True
            )
        self._last_time = time.perf_counter()
        self._tcp_baseline = _tcp_counters()
        self._thread = threading.Thread(
            target=self._run, name="host-counter-sampler", daemon=True
        )
        self._thread.start()
        if self._fast_thread is not None:
            self._fast_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._fast_thread is not None:
            self._fast_thread.join()
            self._fast_thread = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample()
            except Exception:
                logger.exception("Host counter sample failed")

    def _run_fast_net(self) -> None:
        while not self._stop.wait(self._net_interval_s):
            try:
                self._sample_net()
            except Exception:
                logger.exception("Host network burst sample failed")

    def _sample_net(self) -> None:
        now = self._clock()
        net_text = self._net_dev_reader()
        if net_text is None:
            return
        rx_bytes, tx_bytes, _, _ = parse_net_dev(net_text)
        with self._lock:
            elapsed = now - self._fast_last_time
            if self._fast_last is not None and elapsed > 0:
                self._fast_rx_mb_s.append(
                    (rx_bytes - self._fast_last[0]) / elapsed / _MB
                )
                self._fast_tx_mb_s.append(
                    (tx_bytes - self._fast_last[1]) / elapsed / _MB
                )
            self._fast_last = (rx_bytes, tx_bytes)
            self._fast_last_time = now

    def _sample(self) -> None:
        now = time.perf_counter()
        net_text = self._net_dev_reader()
        load_text = _read("/proc/loadavg")
        mem_available = _mem_available_gb()
        rss = _rss_gb(self._pids)
        with self._lock:
            elapsed = now - self._last_time
            self._last_time = now
            if net_text is not None:
                net = parse_net_dev(net_text)
                if self._last_net is not None and elapsed > 0:
                    rx_b, tx_b, rx_p, tx_p = (
                        (cur - prev) / elapsed for cur, prev in zip(net, self._last_net)
                    )
                    self._rx_mb_s.append(rx_b / _MB)
                    self._tx_mb_s.append(tx_b / _MB)
                    self._rx_pkt_s.append(rx_p)
                    self._tx_pkt_s.append(tx_p)
                self._last_net = net
            if load_text is not None:
                self._loadavg.append(float(load_text.split()[0]))
            if mem_available is not None:
                self._mem_available_gb.append(mem_available)
            self._rss_gb.append(rss)
            self._samples += 1

    def make_reporter(self) -> Callable[[], dict[str, float]]:
        """Returns a ``SamplingStats`` reporter summarizing the samples since its last call."""
        return self._report

    def _report(self) -> dict[str, float]:
        with self._lock:
            report: dict[str, float] = {"host_samples": float(self._samples)}
            if self._rx_mb_s:
                report["net_rx_mb_s_mean"] = sum(self._rx_mb_s) / len(self._rx_mb_s)
                report["net_rx_mb_s_max"] = max(self._rx_mb_s)
                report["net_tx_mb_s_mean"] = sum(self._tx_mb_s) / len(self._tx_mb_s)
                report["net_tx_mb_s_max"] = max(self._tx_mb_s)
                report["net_rx_pkt_s_mean"] = sum(self._rx_pkt_s) / len(self._rx_pkt_s)
                report["net_tx_pkt_s_mean"] = sum(self._tx_pkt_s) / len(self._tx_pkt_s)
            for direction, bins in (
                ("rx", self._fast_rx_mb_s),
                ("tx", self._fast_tx_mb_s),
            ):
                if not bins:
                    continue
                threshold = _LINE_RATE_FRACTION * self._line_rate_mb_s
                report[f"net_{direction}_mb_s_p99_100ms"] = _percentile(bins, 0.99)
                report[f"net_{direction}_mb_s_max_100ms"] = max(bins)
                report[f"net_{direction}_bins_over_80pct_line"] = sum(
                    1 for rate in bins if rate > threshold
                ) / len(bins)
            tcp = _tcp_counters()
            for name, key in (
                ("retrans_segs", "tcp_retrans_segs"),
                ("timeouts", "tcp_timeouts"),
                ("loss_probes", "tcp_loss_probes"),
            ):
                if name in tcp:
                    report[key] = float(tcp[name] - self._tcp_baseline.get(name, 0))
            self._tcp_baseline = tcp
            if self._loadavg:
                report["loadavg1_mean"] = sum(self._loadavg) / len(self._loadavg)
            if self._mem_available_gb:
                report["mem_available_gb_min"] = min(self._mem_available_gb)
            if self._rss_gb:
                report["rss_local_workers_gb_max"] = max(self._rss_gb)
            self._reset_window()
            return report
