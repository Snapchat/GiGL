"""Tests for GbmlConfigPbWrapper deprecation behavior of should_run_glt_backend."""

from __future__ import annotations

import warnings

import pytest

from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from snapchat.research.gbml import gbml_config_pb2, graph_schema_pb2


def _build_minimal_gbml_config_pb(
    feature_flags: dict[str, str] | None = None,
) -> gbml_config_pb2.GbmlConfig:
    pb = gbml_config_pb2.GbmlConfig()
    pb.graph_metadata.node_types.append("paper")
    edge_type = graph_schema_pb2.EdgeType(
        relation="cites",
        src_node_type="paper",
        dst_node_type="paper",
    )
    pb.graph_metadata.edge_types.append(edge_type)
    pb.graph_metadata.condensed_node_type_map[0] = "paper"
    pb.graph_metadata.condensed_edge_type_map[0].relation = "cites"
    pb.graph_metadata.condensed_edge_type_map[0].src_node_type = "paper"
    pb.graph_metadata.condensed_edge_type_map[0].dst_node_type = "paper"
    pb.task_metadata.node_based_task_metadata.supervision_node_types.append("paper")
    if feature_flags:
        for k, v in feature_flags.items():
            pb.feature_flags[k] = v
    return pb


def test_unset_flag_defaults_to_true_and_no_warning() -> None:
    pb = _build_minimal_gbml_config_pb(feature_flags=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper = GbmlConfigPbWrapper(gbml_config_pb=pb)
        assert wrapper.should_use_glt_backend is True
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == []


@pytest.mark.parametrize("flag_value", ["True", "False"])
def test_setting_flag_emits_deprecation_warning(flag_value: str) -> None:
    pb = _build_minimal_gbml_config_pb(
        feature_flags={"should_run_glt_backend": flag_value}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapper = GbmlConfigPbWrapper(gbml_config_pb=pb)
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert len(deprecation_warnings) == 1
    assert "should_run_glt_backend" in str(deprecation_warnings[0].message)


def test_explicit_false_returns_false_and_warns() -> None:
    pb = _build_minimal_gbml_config_pb(feature_flags={"should_run_glt_backend": "False"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = GbmlConfigPbWrapper(gbml_config_pb=pb)
        assert wrapper.should_use_glt_backend is False


def test_explicit_true_returns_true() -> None:
    pb = _build_minimal_gbml_config_pb(feature_flags={"should_run_glt_backend": "True"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        wrapper = GbmlConfigPbWrapper(gbml_config_pb=pb)
        assert wrapper.should_use_glt_backend is True
