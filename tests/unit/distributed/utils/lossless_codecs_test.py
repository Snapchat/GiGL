from typing import Optional, cast

import tensorflow as tf
from absl.testing import absltest
from tensorflow_metadata.proto.v0.schema_pb2 import Feature, IntDomain

from gigl.common.data.load_torch_tensors import SerializedGraphMetadata
from gigl.common.utils.feature_quantization.lossless import FeatureCodec
from gigl.distributed.utils.lossless_codecs import (
    codec_from_feature_schema,
    derive_edge_lossless_codecs,
    derive_node_lossless_codecs,
    exclude_quantized_columns,
)
from gigl.distributed.utils.serialized_graph_metadata_translator import (
    convert_pb_to_serialized_graph_metadata,
)
from gigl.src.common.types.graph_data import EdgeType, NodeType, Relation
from gigl.src.common.types.pb_wrappers.gbml_config import GbmlConfigPbWrapper
from gigl.src.common.types.pb_wrappers.preprocessed_metadata import (
    PreprocessedMetadataPbWrapper,
)
from gigl.src.data_preprocessor.lib.types import FeatureSchema
from gigl.src.mocking.lib.versioning import get_mocked_dataset_artifact_metadata
from gigl.src.mocking.mocking_assets.mocked_datasets_for_pipeline_tests import (
    HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
    TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO,
)
from gigl.types.graph import FeatureQuantizationMetadata
from snapchat.research.gbml.preprocessed_metadata_pb2 import PreprocessedMetadata
from tests.test_assets.test_case import TestCase

_USER = NodeType("user")
_STORY = NodeType("story")
_USER_TO_STORY = EdgeType(_USER, Relation("to"), _STORY)


def _schema(*features: tuple[str, int, Optional[tuple[int, int]]]) -> FeatureSchema:
    return FeatureSchema(
        schema={
            name: Feature(int_domain=IntDomain(min=domain[0], max=domain[1]))
            if domain
            else Feature()
            for name, _, domain in features
        },
        feature_spec={
            name: tf.io.FixedLenFeature([width], tf.float32)
            for name, width, _ in features
        },
        feature_index={},
        feature_vocab={},
    )


def _config(mocked_dataset_info) -> GbmlConfigPbWrapper:
    return GbmlConfigPbWrapper.get_gbml_config_pb_wrapper_from_uri(
        get_mocked_dataset_artifact_metadata()[
            mocked_dataset_info.name
        ].frozen_gbml_config_uri
    )


def _serialized_metadata(
    preprocessed_metadata_pb_wrapper: PreprocessedMetadataPbWrapper,
    graph_metadata_pb_wrapper,
) -> SerializedGraphMetadata:
    return convert_pb_to_serialized_graph_metadata(
        preprocessed_metadata_pb_wrapper=preprocessed_metadata_pb_wrapper,
        graph_metadata_pb_wrapper=graph_metadata_pb_wrapper,
    )


class CodecFromFeatureSchemaTest(TestCase):
    def test_columns_follow_the_feature_keys_and_domains(self) -> None:
        schema = _schema(
            ("score", 2, None),
            ("country", 1, (-1, 200)),
            ("topic", 3, (0, 70_000)),
            ("not_listed", 4, (0, 1)),
        )

        codec = codec_from_feature_schema(schema, ["topic", "score", "country"])

        self.assertEqual(
            codec,
            FeatureCodec(("int32",) * 3 + ("float32",) * 2 + ("uint8_bias1",)),
        )

    def test_domain_beyond_int32_is_declared_uncompressed(self) -> None:
        schema = _schema(("huge_ids", 1, (0, 2**40)))
        self.assertEqual(
            codec_from_feature_schema(schema, ["huge_ids"]), FeatureCodec(("float32",))
        )


class DeriveNodeLosslessCodecsTest(TestCase):
    def test_homogeneous_graph_gets_a_single_codec_over_all_logical_columns(
        self,
    ) -> None:
        config = _config(TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO)
        preprocessed = config.preprocessed_metadata_pb_wrapper
        graph = config.graph_metadata_pb_wrapper

        node_codec = derive_node_lossless_codecs(
            preprocessed, graph, _serialized_metadata(preprocessed, graph)
        )

        self.assertEqual(node_codec, FeatureCodec(("float32",) * 2))

    def test_heterogeneous_graph_gets_a_codec_per_node_type(self) -> None:
        config = _config(HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO)

        node_codecs = derive_node_lossless_codecs(
            config.preprocessed_metadata_pb_wrapper,
            config.graph_metadata_pb_wrapper,
            _serialized_metadata(
                config.preprocessed_metadata_pb_wrapper,
                config.graph_metadata_pb_wrapper,
            ),
        )

        self.assertEqual(
            node_codecs,
            {
                _USER: FeatureCodec(("float32",) * 2),
                _STORY: FeatureCodec(("float32",) * 2),
            },
        )

    def test_edge_codecs_cover_the_logical_columns_and_exclude_the_weight_feature(
        self,
    ) -> None:
        config = _config(HETEROGENEOUS_TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO)
        preprocessed = config.preprocessed_metadata_pb_wrapper
        graph = config.graph_metadata_pb_wrapper
        condensed = graph.edge_type_to_condensed_edge_type_map[_USER_TO_STORY]
        weight_name = preprocessed.condensed_edge_type_to_feature_keys_map[condensed][0]

        unweighted = cast(
            dict[EdgeType, FeatureCodec],
            derive_edge_lossless_codecs(
                preprocessed, graph, _serialized_metadata(preprocessed, graph)
            ),
        )
        weighted = cast(
            dict[EdgeType, FeatureCodec],
            derive_edge_lossless_codecs(
                preprocessed,
                graph,
                _serialized_metadata(preprocessed, graph),
                weight_edge_feat_name={_USER_TO_STORY: weight_name},
            ),
        )

        self.assertEqual(unweighted[_USER_TO_STORY].num_columns, 2)
        self.assertEqual(weighted[_USER_TO_STORY].num_columns, 1)
        self.assertEqual(
            {e: c for e, c in weighted.items() if e != _USER_TO_STORY},
            {e: c for e, c in unweighted.items() if e != _USER_TO_STORY},
        )

    def test_graph_without_node_features_gets_no_codec(self) -> None:
        config = _config(TOY_GRAPH_NODE_ANCHOR_MOCKED_DATASET_INFO)
        preprocessed_pb = PreprocessedMetadata()
        preprocessed_pb.CopyFrom(
            config.preprocessed_metadata_pb_wrapper.preprocessed_metadata
        )
        for (
            node_metadata
        ) in preprocessed_pb.condensed_node_type_to_preprocessed_metadata.values():
            node_metadata.ClearField("feature_keys")
            node_metadata.feature_dim = 0

        featureless = PreprocessedMetadataPbWrapper(
            preprocessed_metadata_pb=preprocessed_pb
        )
        self.assertIsNone(
            derive_node_lossless_codecs(
                featureless,
                config.graph_metadata_pb_wrapper,
                _serialized_metadata(featureless, config.graph_metadata_pb_wrapper),
            )
        )


class ExcludeQuantizedColumnsTest(TestCase):
    def test_columns_quantized_in_preprocessing_are_left_uncompressed(self) -> None:
        quantized = FeatureQuantizationMetadata(
            bits=2,
            feature_dim=3,
            quantized_feature_indices=(1,),
            clip_min=0.0,
            clip_max=3.0,
        )
        codec = exclude_quantized_columns(
            FeatureCodec(("int8", "int8", "int16")), quantized
        )
        self.assertEqual(codec, FeatureCodec(("int8", "float32", "int16")))
        self.assertEqual(
            exclude_quantized_columns(FeatureCodec(("int8",)), None),
            FeatureCodec(("int8",)),
        )


if __name__ == "__main__":
    absltest.main()
