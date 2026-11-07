import os
import time

from gigl.distributed.utils.host_stats import (
    HostCounterSampler,
    parse_net_dev,
    parse_netstat_tcpext,
    parse_snmp_tcp,
)
from tests.test_assets.test_case import TestCase

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
            "tcp_retrans_segs",
            "tcp_timeouts",
            "tcp_loss_probes",
            "loadavg1_mean",
            "mem_available_gb_min",
            "rss_pids_gb_max",
        ):
            self.assertIn(key, report)
        self.assertGreater(report["mem_available_gb_min"], 0)
        self.assertGreater(report["rss_pids_gb_max"], 0)

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
