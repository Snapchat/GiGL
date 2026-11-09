import time

from gigl.distributed.utils.flow_stats import FlowStatsSampler, parse_ss_tin
from tests.test_assets.test_case import TestCase

# Two lines per socket: addresses, then an indented TCP info line (as printed by ``ss -tinH``).
_SS_OUTPUT = """\
0      0         10.128.0.4:50922      10.128.0.5:41000
\t cubic wscale:13,7 rto:204 rtt:0.215/0.032 ato:40 mss:8896 pmtu:8940 rcvmss:536 advmss:8896 cwnd:40 bytes_sent:9000000 bytes_acked:8999990 segs_out:1200 send 13.3Gbps lastsnd:4 pacing_rate 26.5Gbps delivery_rate 8.2Gbps delivered:1000 busy:900ms rwnd_limited:100ms(11.1%) sndbuf_limited:20ms(2.2%) retrans:0/3 minrtt:0.1
0      0         10.128.0.4:50924      10.128.0.5:41001
\t cubic wscale:13,7 rto:204 rtt:0.4/0.1 mss:8896 cwnd:10 bytes_acked:5000 send 1.78Gbps delivery_rate 250Mbps busy:10ms
0      0         10.128.0.4:50926      10.128.0.5:41002
\t cubic rto:204 rtt:0.3/0.1 cwnd:10 send 5000000bps delivery_rate 800000bps
0      0         10.128.0.4:50928      10.128.0.5:41003
\t cubic rto:204 rtt:1.5/0.1 cwnd:10 bytes_acked:77 send 600Kbps
0      0         127.0.0.1:50930       127.0.0.1:6379
\t cubic rtt:0.05/0.01 cwnd:10 delivery_rate 30Gbps
0      0         10.128.0.4:51524      169.254.169.254:80
\t cubic rtt:0.136/0.037 cwnd:10 delivery_rate 167058816bps
"""


class FlowStatsParserTest(TestCase):
    def test_parses_two_line_sockets_and_units(self) -> None:
        flows = parse_ss_tin(_SS_OUTPUT)
        self.assertEqual(len(flows), 6)
        first = flows[0]
        self.assertEqual(first.local, "10.128.0.4:50922")
        self.assertAlmostEqual(first.rtt_ms or 0, 0.215)
        self.assertEqual(first.cwnd, 40)
        self.assertAlmostEqual(first.delivery_rate_mbps or 0, 8200.0)
        self.assertAlmostEqual(first.send_mbps or 0, 13300.0)
        self.assertEqual(first.retrans_total, 3)
        self.assertEqual(first.bytes_sent, 9000000)
        self.assertEqual(first.bytes_acked, 8999990)
        self.assertAlmostEqual(first.busy_ms or 0, 900.0)
        self.assertAlmostEqual(first.rwnd_limited_ms or 0, 100.0)
        self.assertAlmostEqual(first.sndbuf_limited_ms or 0, 20.0)
        self.assertAlmostEqual(flows[1].delivery_rate_mbps or 0, 250.0)
        self.assertAlmostEqual(flows[2].delivery_rate_mbps or 0, 0.8)
        self.assertAlmostEqual(flows[2].send_mbps or 0, 5.0)

    def test_line_missing_delivery_rate(self) -> None:
        flow = parse_ss_tin(_SS_OUTPUT)[3]
        self.assertIsNone(flow.delivery_rate_mbps)
        self.assertAlmostEqual(flow.send_mbps or 0, 0.6)
        self.assertIsNone(flow.busy_ms)

    def test_accepts_header_and_state_column(self) -> None:
        text = (
            "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
            "ESTAB 0 0 10.0.0.1:1 10.0.0.2:2\n"
            "\t cubic rtt:2/1 cwnd:5 delivery_rate 3Mbps\n"
        )
        (flow,) = parse_ss_tin(text)
        self.assertEqual(flow.peer, "10.0.0.2:2")
        self.assertAlmostEqual(flow.delivery_rate_mbps or 0, 3.0)


class FlowStatsSamplerTest(TestCase):
    def test_report_filters_and_summarizes(self) -> None:
        sampler = FlowStatsSampler(interval_s=1, ss_runner=lambda: _SS_OUTPUT)
        sampler.sample()
        sampler.sample()
        report = sampler.make_reporter()()
        self.assertEqual(report["flow_samples"], 2)
        # Loopback and metadata flows are excluded: 4 data flows, 2 with delivery_rate > 1 Mbps.
        self.assertEqual(report["flows_total"], 4)
        self.assertEqual(report["flows_active"], 2)
        self.assertAlmostEqual(report["flow_delivery_rate_mbps_max"], 8200.0)
        self.assertAlmostEqual(report["flow_delivery_rate_mbps_p50"], 4225.0)
        self.assertAlmostEqual(report["flow_rtt_ms_p50"], 0.35)
        self.assertEqual(report["flow_retrans_total"], 3)
        self.assertEqual(report["flows_rwnd_limited_frac"], 0.5)
        self.assertEqual(report["flows_sndbuf_limited_frac"], 0.5)
        self.assertEqual(sampler.make_reporter()()["flow_samples"], 0)

    def test_background_thread_samples(self) -> None:
        sampler = FlowStatsSampler(interval_s=0.05, ss_runner=lambda: _SS_OUTPUT)
        sampler.start()
        try:
            time.sleep(0.3)
            report = sampler.make_reporter()()
        finally:
            sampler.stop()
        self.assertGreaterEqual(report["flow_samples"], 2)

    def test_real_ss_on_this_host(self) -> None:
        sampler = FlowStatsSampler(interval_s=0.05)
        sampler.start()
        try:
            time.sleep(0.5)
            report = sampler.make_reporter()()
        finally:
            sampler.stop()
        self.assertIn("flow_samples", report)
