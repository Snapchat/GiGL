import os
import time

from gigl.distributed.utils.host_stats import (
    HostCounterSampler,
    parse_net_dev,
    parse_netstat_tcpext,
    parse_snmp_tcp,
)
from tests.test_assets.test_case import TestCase

_MB = 2**20
_NET_DEV = """Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo: 1000 10 0 0 0 0 0 0 1000 10 0 0 0 0 0 0
  eth0: 2000 20 0 0 0 0 0 0 3000 30 0 0 0 0 0 0
  eth1: 400 4 0 0 0 0 0 0 600 6 0 0 0 0 0 0
"""

_SNMP = """Ip: Forwarding DefaultTTL
Ip: 1 64
Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens RetransSegs InErrs
Tcp: 1 200 120000 -1 5 77 0
Udp: InDatagrams NoPorts
Udp: 9 0
"""

_NETSTAT = """TcpExt: SyncookiesSent TCPTimeouts TCPLossProbes
TcpExt: 0 12 34
IpExt: InNoRoutes
IpExt: 0
"""


class ParseTest(TestCase):
    def test_parse_net_dev_sums_non_loopback_interfaces(self) -> None:
        self.assertEqual(parse_net_dev(_NET_DEV), (2400, 3600, 24, 36))

    def test_parse_snmp_tcp(self) -> None:
        tcp = parse_snmp_tcp(_SNMP)
        self.assertEqual(tcp["RetransSegs"], 77)
        self.assertEqual(tcp["MaxConn"], -1)

    def test_parse_netstat_tcpext(self) -> None:
        self.assertEqual(
            parse_netstat_tcpext(_NETSTAT),
            {"SyncookiesSent": 0, "TCPTimeouts": 12, "TCPLossProbes": 34},
        )


class HostCounterSamplerTest(TestCase):
    def test_sampler_reports_gauges_and_resets_per_window(self) -> None:
        sampler = HostCounterSampler(interval_s=0.05, pids=[os.getpid()])
        sampler.start()
        try:
            time.sleep(0.3)
            report = sampler.make_reporter()()
        finally:
            sampler.stop()
        self.assertGreaterEqual(report["host_samples"], 3)
        for key in (
            "net_rx_mb_s_mean",
            "net_rx_mb_s_max",
            "net_tx_mb_s_mean",
            "net_tx_mb_s_max",
            "net_rx_pkt_s_mean",
            "net_tx_pkt_s_mean",
            "net_rx_mb_s_p99_100ms",
            "net_rx_mb_s_max_100ms",
            "net_tx_mb_s_p99_100ms",
            "net_tx_mb_s_max_100ms",
            "net_rx_bins_over_80pct_line",
            "net_tx_bins_over_80pct_line",
            "tcp_retrans_segs",
            "tcp_timeouts",
            "tcp_loss_probes",
            "loadavg1_mean",
            "mem_available_gb_min",
            "rss_local_workers_gb_max",
        ):
            self.assertIn(key, report)
        self.assertGreater(report["mem_available_gb_min"], 0)
        self.assertGreater(report["rss_local_workers_gb_max"], 0)

    def test_second_reporter_call_is_a_fresh_window(self) -> None:
        sampler = HostCounterSampler(interval_s=0.05, pids=[os.getpid()])
        sampler.start()
        try:
            time.sleep(0.3)
            reporter = sampler.make_reporter()
            first = reporter()
            second = reporter()
        finally:
            sampler.stop()
        self.assertGreaterEqual(first["host_samples"], 3)
        self.assertLess(second["host_samples"], first["host_samples"])


def _net_dev(rx_bytes: int, tx_bytes: int) -> str:
    return f"  eth0: {rx_bytes} 1 0 0 0 0 0 0 {tx_bytes} 1 0 0 0 0 0 0\n"


class BurstBinTest(TestCase):
    def test_100ms_bins_see_a_burst_the_2s_mean_hides(self) -> None:
        # Ten 0.1 s bins: nine at 100 MB/s rx and one at 5000 MB/s rx. Tx is a steady 10 MB/s.
        rx_rates = [100] * 9 + [5000]
        counters = [_net_dev(0, 0)]
        rx = 0
        tx = 0
        for rate in rx_rates:
            rx += int(rate * _MB * 0.1)
            tx += int(10 * _MB * 0.1)
            counters.append(_net_dev(rx, tx))
        current = [counters[0]]
        now = [0.0]

        # The background threads' interval is long so only the explicit samples below count.
        sampler = HostCounterSampler(
            interval_s=3600,
            net_interval_s=3600,
            net_dev_reader=lambda: current[0],
            clock=lambda: now[0],
        )
        sampler.start()
        try:
            for text in counters[1:]:
                now[0] += 0.1
                current[0] = text
                sampler._sample_net()
            report = sampler.make_reporter()()
        finally:
            sampler.stop()

        self.assertAlmostEqual(report["net_rx_mb_s_max_100ms"], 5000, delta=1)
        # Linear interpolation: 0.99 * 9 = 8.91 -> 100 + 0.91 * 4900.
        self.assertAlmostEqual(report["net_rx_mb_s_p99_100ms"], 4559, delta=5)
        self.assertAlmostEqual(report["net_tx_mb_s_max_100ms"], 10, delta=0.1)
        # 80% of the default 50 Gbps line is about 4768 MiB/s, so only the burst bin is over.
        self.assertAlmostEqual(report["net_rx_bins_over_80pct_line"], 0.1)
        self.assertEqual(report["net_tx_bins_over_80pct_line"], 0.0)
