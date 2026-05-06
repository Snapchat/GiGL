"""Unit tests for Pydantic result types and artifact IO."""
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from gigl.analytics.data_analyzer.types import (
    SCHEMA_VERSION,
    DegreeStats,
    EmbeddingDiagnosticsResult,
    FeatureProfileArtifact,
    FeatureProfileError,
    FeatureProfileResult,
    GraphAnalysisResult,
    GraphStructureArtifact,
    LabelSentinelStats,
    NodeClassificationSupervisionStats,
    PerClassDegreeStats,
    SupervisionCrossTableStats,
    TopKEntry,
    load_artifact,
    write_artifact,
)
from tests.test_assets.test_case import TestCase


class SchemaVersionTest(TestCase):
    def test_schema_version_is_one(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "1")


class ResultRoundtripTest(TestCase):
    def test_graph_analysis_result_roundtrip(self) -> None:
        original = GraphAnalysisResult(
            node_counts={"user": 1000},
            edge_counts={"follows": 5000},
            degree_stats={
                "follows_out": DegreeStats(
                    min=0,
                    max=100,
                    mean=5.0,
                    median=3,
                    p90=20,
                    p99=50,
                    p999=80,
                    percentiles=list(range(101)),
                    buckets={"0-1": 100},
                )
            },
            top_hubs={"follows_out": [("u1", 50), ("u2", 40)]},
        )
        serialized = original.model_dump_json()
        rehydrated = GraphAnalysisResult.model_validate_json(serialized)
        self.assertEqual(rehydrated, original)

    def test_feature_profile_errors_roundtrip(self) -> None:
        original = FeatureProfileResult(
            errors=[
                FeatureProfileError(
                    result_key="node:user",
                    bq_table="p.d.users",
                    stage="schema_fetch",
                    message="permission denied",
                ),
                FeatureProfileError(
                    result_key="edge:follows",
                    bq_table="p.d.follows",
                    stage="dataflow",
                    message="RuntimeError: boom",
                ),
            ],
        )
        serialized = original.model_dump_json()
        rehydrated = FeatureProfileResult.model_validate_json(serialized)
        self.assertEqual(rehydrated, original)

    def test_feature_profile_result_roundtrip(self) -> None:
        original = FeatureProfileResult(
            facets_html_paths={"node:user": ["gs://b/facets.html"]},
            stats_paths={"node:user": ["gs://b/stats.tfrecord"]},
            embedding_diagnostics={
                "node:user": {
                    "emb": EmbeddingDiagnosticsResult(
                        total=100,
                        unique_count=98,
                        unique_ratio=0.98,
                        top_k=[TopKEntry(hash=42, count=2, fraction=0.02)],
                    )
                }
            },
        )
        serialized = original.model_dump_json()
        rehydrated = FeatureProfileResult.model_validate_json(serialized)
        self.assertEqual(rehydrated, original)

    def test_supervision_cross_table_stats_roundtrip(self) -> None:
        original = GraphAnalysisResult(
            supervision_cross_table_stats=[
                SupervisionCrossTableStats(
                    driver_edge_type="viewed_pos",
                    driver_role="supervision_pos",
                    other_edge_type="viewed_neg",
                    other_role="supervision_neg",
                    node_anchor="user",
                    driver_anchor_count=1000,
                    driver_pair_count=5000,
                    other_pair_count=6000,
                    overlap_pair_count=3,
                    driver_anchors_with_zero_other=50,
                    avg_other_per_driver_anchor=4.5,
                    p50_other_per_driver_anchor=4,
                    p90_other_per_driver_anchor=12,
                    p99_other_per_driver_anchor=40,
                    max_other_per_driver_anchor=200,
                )
            ],
        )
        serialized = original.model_dump_json()
        rehydrated = GraphAnalysisResult.model_validate_json(serialized)
        self.assertEqual(rehydrated, original)

    def test_per_class_degree_buckets_roundtrip(self) -> None:
        """Per-class buckets, sentinel degree stats, and the queries log all
        survive JSON round-trip.
        """
        original = GraphAnalysisResult(
            node_classification_supervision_stats=[
                NodeClassificationSupervisionStats(
                    node_type="user",
                    label_column="label",
                    sentinel_stats=LabelSentinelStats(
                        total_rows=10,
                        null_count=0,
                        valid_label_count=10,
                        valid_label_coverage=1.0,
                    ),
                    per_class_degree=[
                        PerClassDegreeStats(
                            class_value="0",
                            count=600,
                            cold_start_count=30,
                            mean_degree=5.0,
                            median_degree=4,
                            p90_degree=20,
                            p99_degree=80,
                            max_degree=100,
                            buckets={
                                "0-1": 30,
                                "2-10": 500,
                                "11-100": 60,
                                "101-1K": 10,
                                "1K-10K": 0,
                                "10K+": 0,
                            },
                        )
                    ],
                    sentinel_degree_stats=[
                        PerClassDegreeStats(
                            class_value="-1",
                            count=50,
                            cold_start_count=40,
                            mean_degree=1.5,
                            median_degree=1,
                            p90_degree=5,
                            p99_degree=40,
                            max_degree=80,
                            buckets={
                                "0-1": 40,
                                "2-10": 8,
                                "11-100": 2,
                                "101-1K": 0,
                                "1K-10K": 0,
                                "10K+": 0,
                            },
                        )
                    ],
                )
            ],
            queries={
                "nc_supervision:per_class_degree:user": ["SELECT 1"],
                "graph_structure:degree:follows_out": ["SELECT 2", "SELECT 3"],
            },
        )
        serialized = original.model_dump_json()
        rehydrated = GraphAnalysisResult.model_validate_json(serialized)
        self.assertEqual(rehydrated, original)
        self.assertEqual(
            rehydrated.queries["graph_structure:degree:follows_out"],
            ["SELECT 2", "SELECT 3"],
        )
        self.assertEqual(
            rehydrated.node_classification_supervision_stats[0]
            .sentinel_degree_stats[0]
            .class_value,
            "-1",
        )

    def test_extra_fields_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraphAnalysisResult.model_validate(
                {"node_counts": {"user": 1}, "unknown_field": 42}
            )


class WriteArtifactTest(TestCase):
    def test_writes_versioned_envelope_locally(self) -> None:
        result = GraphAnalysisResult(node_counts={"user": 1})
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=result,
                component="graph_structure",
                output_gcs_path=tmp,
            )
            self.assertTrue(path.endswith("/graph_structure.json"))
            payload = json.loads(Path(path).read_text())
            self.assertEqual(payload["schema_version"], "1")
            self.assertEqual(payload["component"], "graph_structure")
            self.assertIn("generated_at", payload)
            self.assertEqual(payload["data"]["node_counts"], {"user": 1})

    def test_feature_profile_component_writes_correct_name(self) -> None:
        result = FeatureProfileResult(
            facets_html_paths={"node:user": ["x"]},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=result,
                component="feature_profile",
                output_gcs_path=tmp,
            )
            self.assertTrue(path.endswith("/feature_profile.json"))

    def test_type_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(TypeError):
                write_artifact(
                    result=FeatureProfileResult(),  # wrong component pairing
                    component="graph_structure",
                    output_gcs_path=tmp,
                )

    def test_trailing_slash_normalized(self) -> None:
        result = GraphAnalysisResult()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=result,
                component="graph_structure",
                output_gcs_path=tmp + "/",
            )
            # The final path should be `tmp/graph_structure.json`, not
            # `tmp//graph_structure.json`.
            self.assertNotIn("//", path.replace("file://", ""))

    def test_creates_parent_directory_if_missing(self) -> None:
        result = GraphAnalysisResult()
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "nested" / "dir"
            path = write_artifact(
                result=result,
                component="graph_structure",
                output_gcs_path=str(nested),
            )
            self.assertTrue(Path(path).exists())


class LoadArtifactTest(TestCase):
    def test_round_trip_via_write_then_load(self) -> None:
        original = GraphAnalysisResult(node_counts={"user": 1000})
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=original,
                component="graph_structure",
                output_gcs_path=tmp,
            )
            loaded = load_artifact(path, expected_component="graph_structure")
        self.assertEqual(loaded, original)

    def test_feature_profile_round_trip(self) -> None:
        original = FeatureProfileResult(
            facets_html_paths={"node:user": ["gs://b/facets.html"]},
            embedding_diagnostics={
                "node:user": {
                    "emb": EmbeddingDiagnosticsResult(
                        total=100,
                        unique_count=100,
                        unique_ratio=1.0,
                    )
                }
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=original,
                component="feature_profile",
                output_gcs_path=tmp,
            )
            loaded = load_artifact(path, expected_component="feature_profile")
        self.assertEqual(loaded, original)

    def test_load_mismatched_component_raises(self) -> None:
        result = FeatureProfileResult()
        with tempfile.TemporaryDirectory() as tmp:
            path = write_artifact(
                result=result,
                component="feature_profile",
                output_gcs_path=tmp,
            )
            with self.assertRaises(ValidationError):
                # Loader expected graph_structure but file is feature_profile.
                load_artifact(path, expected_component="graph_structure")


class EnvelopeValidationTest(TestCase):
    def test_schema_version_literal_is_enforced(self) -> None:
        now = datetime.now(timezone.utc)
        envelope = GraphStructureArtifact(generated_at=now, data=GraphAnalysisResult())
        self.assertEqual(envelope.schema_version, "1")
        self.assertEqual(envelope.component, "graph_structure")

    def test_wrong_component_discriminator_is_rejected(self) -> None:
        # GraphStructureArtifact has component=Literal["graph_structure"]; a
        # JSON blob with component="feature_profile" must fail validation.
        payload = {
            "schema_version": "1",
            "component": "feature_profile",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": {},
        }
        with self.assertRaises(ValidationError):
            GraphStructureArtifact.model_validate(payload)

    def test_feature_profile_envelope_carries_embedding_diagnostics(self) -> None:
        envelope = FeatureProfileArtifact(
            generated_at=datetime.now(timezone.utc),
            data=FeatureProfileResult(
                embedding_diagnostics={
                    "node:user": {
                        "emb": EmbeddingDiagnosticsResult(
                            total=1,
                            unique_count=1,
                            unique_ratio=1.0,
                        )
                    }
                }
            ),
        )
        serialized = envelope.model_dump_json()
        rehydrated = FeatureProfileArtifact.model_validate_json(serialized)
        self.assertEqual(rehydrated, envelope)
