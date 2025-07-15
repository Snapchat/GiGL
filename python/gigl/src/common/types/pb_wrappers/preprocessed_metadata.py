from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List

from tensorflow_metadata.proto.v0.schema_pb2 import Schema

import gigl.common.utils.local_fs as LocalFsUtils
from gigl.common import GcsUri, LocalUri, Uri, UriFactory
from gigl.common.collections.sorted_dict import SortedDict
from gigl.common.logger import Logger
from gigl.common.utils.gcs import GcsUtils
from gigl.common.utils.tensorflow_schema import (
    feature_spec_to_feature_index_map,
    load_tf_schema_uri_str_to_feature_spec,
)
from gigl.src.common.types.graph_data import CondensedEdgeType, CondensedNodeType
from gigl.src.data_preprocessor.lib.types import (
    FeatureSchema,
    FeatureSchemaDict,
    FeatureSpecDict,
    FeatureVocabDict,
)
from snapchat.research.gbml import preprocessed_metadata_pb2

logger = Logger()


@dataclass(frozen=True)
class PreprocessedMetadataPbWrapper:
    preprocessed_metadata_pb: preprocessed_metadata_pb2.PreprocessedMetadata

    _condensed_node_type_to_feature_dim_map: Dict[CondensedNodeType, int] = field(
        init=False
    )
    _condensed_node_type_to_feature_schema_map: Dict[
        CondensedNodeType, FeatureSchema
    ] = field(init=False)

    _condensed_edge_type_to_feature_dim_map: Dict[CondensedEdgeType, int] = field(
        init=False
    )
    _condensed_edge_type_to_feature_schema_map: Dict[
        CondensedEdgeType, FeatureSchema
    ] = field(init=False)

    _condensed_edge_type_to_pos_edge_feature_dim_map: Dict[
        CondensedEdgeType, int
    ] = field(init=False)
    _condensed_edge_type_to_pos_edge_feature_schema_map: Dict[
        CondensedEdgeType, FeatureSchema
    ] = field(init=False)

    _condensed_edge_type_to_hard_neg_edge_feature_dim_map: Dict[
        CondensedEdgeType, int
    ] = field(init=False)
    _condensed_edge_type_to_hard_neg_edge_feature_schema_map: Dict[
        CondensedEdgeType, FeatureSchema
    ] = field(init=False)

    def __post_init__(self):
        # Populate the _condensed_node_type_to_feature_dim_map field
        condensed_node_type_to_feature_dim_map: Dict[CondensedNodeType, int] = {}
        condensed_node_type_to_feature_schema_map: Dict[
            CondensedNodeType, FeatureSchema
        ] = {}
        for condensed_node_type, node_metadata in dict(
            self.preprocessed_metadata_pb.condensed_node_type_to_preprocessed_metadata
        ).items():
            condensed_node_type_to_feature_dim_map[
                CondensedNodeType(condensed_node_type)
            ] = node_metadata.feature_dim
            node_feature_keys = list(node_metadata.feature_keys)
            node_feature_schema = self.__build_feature_schema(
                schema_uri=UriFactory.create_uri(node_metadata.schema_uri),
                transform_fn_assets_uri=UriFactory.create_uri(
                    node_metadata.transform_fn_assets_uri
                ),
                feature_keys=node_feature_keys,
            )
            condensed_node_type_to_feature_schema_map[
                CondensedNodeType(condensed_node_type)
            ] = node_feature_schema

        object.__setattr__(
            self,
            "_condensed_node_type_to_feature_dim_map",
            SortedDict(condensed_node_type_to_feature_dim_map),
        )
        object.__setattr__(
            self,
            "_condensed_node_type_to_feature_schema_map",
            SortedDict(condensed_node_type_to_feature_schema_map),
        )

        # Populate the _condensed_edge_type_to_feature_dim_map field
        condensed_edge_type_to_feature_dim_map: Dict[CondensedEdgeType, int] = {}
        condensed_edge_type_to_feature_schema_map: Dict[
            CondensedEdgeType, FeatureSchema
        ] = {}
        for condensed_edge_type, edge_metadata in dict(
            self.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata
        ).items():
            main_edge_info = edge_metadata.main_edge_info
            condensed_edge_type_to_feature_dim_map[
                CondensedEdgeType(condensed_edge_type)
            ] = main_edge_info.feature_dim
            main_edge_feature_keys = list(main_edge_info.feature_keys)
            edge_feature_schema = self.__build_feature_schema(
                schema_uri=UriFactory.create_uri(main_edge_info.schema_uri),
                transform_fn_assets_uri=UriFactory.create_uri(
                    main_edge_info.transform_fn_assets_uri
                ),
                feature_keys=main_edge_feature_keys,
            )
            condensed_edge_type_to_feature_schema_map[
                CondensedEdgeType(condensed_edge_type)
            ] = edge_feature_schema
        object.__setattr__(
            self,
            "_condensed_edge_type_to_feature_dim_map",
            SortedDict(condensed_edge_type_to_feature_dim_map),
        )
        object.__setattr__(
            self,
            "_condensed_edge_type_to_feature_schema_map",
            SortedDict(condensed_edge_type_to_feature_schema_map),
        )

        # Populate the _condensed_edge_type_to_pos_edge_feature_dim_map field
        condensed_edge_type_to_pos_edge_feature_dim_map: Dict[
            CondensedEdgeType, int
        ] = {}
        condensed_edge_type_to_pos_edge_feature_schema_map: Dict[
            CondensedEdgeType, FeatureSchema
        ] = {}
        for condensed_edge_type, edge_metadata in dict(
            self.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata
        ).items():
            pos_edge_info = edge_metadata.positive_edge_info
            condensed_edge_type_to_pos_edge_feature_dim_map[
                CondensedEdgeType(condensed_edge_type)
            ] = pos_edge_info.feature_dim
            pos_edge_feature_keys = list(pos_edge_info.feature_keys)
            pos_edge_feature_schema = self.__build_feature_schema(
                schema_uri=UriFactory.create_uri(pos_edge_info.schema_uri),
                transform_fn_assets_uri=UriFactory.create_uri(
                    pos_edge_info.transform_fn_assets_uri
                ),
                feature_keys=pos_edge_feature_keys,
            )
            condensed_edge_type_to_pos_edge_feature_schema_map[
                CondensedEdgeType(condensed_edge_type)
            ] = pos_edge_feature_schema
        object.__setattr__(
            self,
            "_condensed_edge_type_to_pos_edge_feature_dim_map",
            SortedDict(condensed_edge_type_to_pos_edge_feature_dim_map),
        )
        object.__setattr__(
            self,
            "_condensed_edge_type_to_pos_edge_feature_schema_map",
            SortedDict(condensed_edge_type_to_pos_edge_feature_schema_map),
        )

        # Populate the _condensed_edge_type_to_hard_neg_edge_feature_dim_map field
        condensed_edge_type_to_hard_neg_edge_feature_dim_map: Dict[
            CondensedEdgeType, int
        ] = {}
        condensed_edge_type_to_hard_neg_edge_feature_schema_map: Dict[
            CondensedEdgeType, FeatureSchema
        ] = {}
        for condensed_edge_type, edge_metadata in dict(
            self.preprocessed_metadata_pb.condensed_edge_type_to_preprocessed_metadata
        ).items():
            hard_neg_edge_info = edge_metadata.negative_edge_info
            condensed_edge_type_to_hard_neg_edge_feature_dim_map[
                CondensedEdgeType(condensed_edge_type)
            ] = hard_neg_edge_info.feature_dim
            hard_neg_edge_feature_keys = list(hard_neg_edge_info.feature_keys)
            hard_neg_edge_feature_schema = self.__build_feature_schema(
                schema_uri=UriFactory.create_uri(hard_neg_edge_info.schema_uri),
                transform_fn_assets_uri=UriFactory.create_uri(
                    hard_neg_edge_info.transform_fn_assets_uri
                ),
                feature_keys=hard_neg_edge_feature_keys,
            )
            condensed_edge_type_to_hard_neg_edge_feature_schema_map[
                CondensedEdgeType(condensed_edge_type)
            ] = hard_neg_edge_feature_schema
        object.__setattr__(
            self,
            "_condensed_edge_type_to_hard_neg_edge_feature_dim_map",
            SortedDict(condensed_edge_type_to_hard_neg_edge_feature_dim_map),
        )
        object.__setattr__(
            self,
            "_condensed_edge_type_to_hard_neg_edge_feature_schema_map",
            SortedDict(condensed_edge_type_to_hard_neg_edge_feature_schema_map),
        )

    def __get_feature_spec_for_feature_keys(
        self, feature_spec: FeatureSpecDict, feature_keys: List[str]
    ) -> FeatureSpecDict:
        """
        Return feature spec for the given feature keys
        """
        return {feature: feature_spec[feature] for feature in feature_keys}

    def __get_schema_for_feature_keys(
        self,
        feature_schema: Schema,
        feature_spec: FeatureSpecDict,
        feature_keys: List[str],
    ) -> FeatureSchemaDict:
        """
        Return feature schema for the given feature keys
        """
        all_features_in_feature_spec = list(feature_spec.keys())
        return {
            feature: feature_schema.feature[all_features_in_feature_spec.index(feature)]
            for feature in feature_keys
        }

    def __build_feature_schema(
        self,
        schema_uri: Uri,
        transform_fn_assets_uri: Uri,
        feature_keys: List[str],
    ) -> FeatureSchema:
        """
        Return FeatureSchema NamedTuple for the given feature keys
        that includes the tf schema, feature spec, and feature index (start, end)

        feature_spec can also contain id keys or label keys, we only need the feature keys
        feature_spec will be based on the order of feature keys in preprocessed metadata
        which is also how SGS processes the features currently into float vectors
        """
        raw_feature_schema, raw_feature_spec = (
            load_tf_schema_uri_str_to_feature_spec(uri=schema_uri)
            if schema_uri.uri
            else (None, {})
        )

        feature_spec = (
            self.__get_feature_spec_for_feature_keys(
                feature_spec=raw_feature_spec, feature_keys=feature_keys
            )
            if feature_keys and raw_feature_schema
            else {}
        )
        schema = (
            self.__get_schema_for_feature_keys(
                feature_schema=raw_feature_schema,
                feature_spec=raw_feature_spec,
                feature_keys=feature_keys,
            )
            if feature_keys and raw_feature_schema
            else {}
        )
        feature_vocab = (
            self.__get_feature_to_vocab_list_map(
                transform_fn_assets_uri=transform_fn_assets_uri
            )
            if transform_fn_assets_uri.uri
            else {}
        )
        return FeatureSchema(
            schema=schema,
            feature_spec=feature_spec,
            feature_index=feature_spec_to_feature_index_map(feature_spec),
            feature_vocab=feature_vocab,
        )

    def __get_feature_to_vocab_list_map(
        self,
        transform_fn_assets_uri: Uri,
    ) -> FeatureVocabDict:
        if isinstance(transform_fn_assets_uri, LocalUri):
            list_files_fn = partial(LocalFsUtils.list_at_path, entity=LocalFsUtils.FileSystemEntity.FILE)  # type: ignore
            read_file_fn = lambda path: open(path, "rb")  # type: ignore
        elif isinstance(transform_fn_assets_uri, GcsUri):
            gcs_utils = GcsUtils()
            list_files_fn = gcs_utils.list_uris_with_gcs_path_pattern  # type: ignore
            read_file_fn = gcs_utils.download_file_from_gcs_to_temp_file  # type: ignore
        else:
            raise ValueError(
                f"Invalid uri: {transform_fn_assets_uri}. Must be either {GcsUri.__name__} or {LocalUri.__name__}"
            )

        assets_file_paths = list_files_fn(transform_fn_assets_uri)  # type: ignore
        feature_to_vocab_list_map = {}
        for asset_file_path in assets_file_paths:
            feature_key = asset_file_path.uri.split("/")[-1]
            f = read_file_fn(asset_file_path)
            vocab_list = [line.decode().rstrip() for line in f]
            feature_to_vocab_list_map[feature_key] = vocab_list
            f.close()

        return feature_to_vocab_list_map

    def has_pos_edge_features(self, condensed_edge_type: CondensedEdgeType) -> bool:
        """
        Returns whether the given CondensedEdgeType has positive edge features.

        Args:
            condensed_edge_type (CondensedEdgeType): The CondensedEdgeType to check

        Returns:
            bool: Whether the given CondensedEdgeType has positive edge features
        """
        return (
            self._condensed_edge_type_to_pos_edge_feature_dim_map[condensed_edge_type]
            > 0
        )

    def has_hard_neg_edge_features(
        self, condensed_edge_type: CondensedEdgeType
    ) -> bool:
        """
        Returns whether the given CondensedEdgeType has hard negative edge features.

        Args:
            condensed_edge_type (CondensedEdgeType): The CondensedEdgeType to check

        Returns:
            bool: Whether the given CondensedEdgeType has hard negative edge features
        """
        return (
            self._condensed_edge_type_to_hard_neg_edge_feature_dim_map[
                condensed_edge_type
            ]
            > 0
        )

    @property
    def preprocessed_metadata(self) -> preprocessed_metadata_pb2.PreprocessedMetadata:
        """
        Allows access to the underlying PreprocessedMetadata protobuf.

        Returns:
            preprocessed_metadata_pb2.PreprocessedMetadata: The underlying PreprocessedMetadata protobuf
        """
        return self.preprocessed_metadata_pb

    @property
    def condensed_node_type_to_feature_dim_map(
        self,
    ) -> Dict[CondensedNodeType, int]:
        """
        Allows access to a mapping which stores the feature dimension of each
        CondensedNodeTypes.

        Returns:
            Dict[CondensedNodeType, int]: A mapping which stores the feature dimension of each
            CondensedNodeTypes
        """
        return self._condensed_node_type_to_feature_dim_map

    @property
    def condensed_node_type_to_feature_schema_map(
        self,
    ) -> Dict[CondensedNodeType, FeatureSchema]:
        """
        Allows access to a mapping which stores the feature spec of each
        CondensedNodeTypes.

        Returns:
            Dict[CondensedNodeType, FeatureSchema]: A mapping which stores the feature spec of each
            CondensedNodeTypes
        """
        return self._condensed_node_type_to_feature_schema_map

    @property
    def condensed_node_type_to_feature_keys_map(
        self,
    ) -> Dict[CondensedNodeType, List[str]]:
        """
        Allows access to a mapping which stores the feature keys of each CondensedNodeTypes.

        Returns:
            Dict[CondensedNodeType, List[str]]: A mapping which stores the feature keys of each CondensedNodeTypes
        """
        return {
            condensed_node_type: list(
                self.condensed_node_type_to_feature_schema_map[
                    condensed_node_type
                ].feature_spec.keys()
            )
            for condensed_node_type in self.condensed_node_type_to_feature_schema_map
        }

    @property
    def condensed_edge_type_to_feature_dim_map(
        self,
    ) -> Dict[CondensedEdgeType, int]:
        """
        Allows access to a mapping which stores the message passing edge feature
        dimension of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, int]: A mapping which stores the message passing edge feature
            dimension of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_feature_dim_map

    @property
    def condensed_edge_type_to_feature_schema_map(
        self,
    ) -> Dict[CondensedEdgeType, FeatureSchema]:
        """
        Allows access to a mapping which stores the message passing edge feature
        spec, tf schema, and feature index of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, FeatureSchema]: A mapping which stores the message passing edge feature
            spec, tf schema, and feature index of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_feature_schema_map

    @property
    def condensed_edge_type_to_feature_keys_map(
        self,
    ) -> Dict[CondensedEdgeType, List[str]]:
        """
        Allows access to a mapping which stores the feature keys of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, List[str]]: A mapping which stores the feature keys of each CondensedEdgeTypes
        """
        return {
            condensed_edge_type: list(
                self.condensed_edge_type_to_feature_schema_map[
                    condensed_edge_type
                ].feature_spec.keys()
            )
            for condensed_edge_type in self.condensed_edge_type_to_feature_schema_map
        }

    @property
    def condensed_edge_type_to_pos_edge_feature_dim_map(
        self,
    ) -> Dict[CondensedEdgeType, int]:
        """
        Allows access to a mapping which stores the user-defined positive edge feature
        dimension of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, int]: A mapping which stores the user-defined positive edge feature
            dimension of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_pos_edge_feature_dim_map

    @property
    def condensed_edge_type_to_pos_edge_feature_schema_map(
        self,
    ) -> Dict[CondensedEdgeType, FeatureSchema]:
        """
        Allows access to a mapping which stores the user-defined positive edge feature
        spec, tf schema, and feature index of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, FeatureSchema]: A mapping which stores the user-defined positive edge feature
            spec, tf schema, and feature index of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_pos_edge_feature_schema_map

    @property
    def condensed_edge_type_to_pos_edge_feature_keys_map(
        self,
    ) -> Dict[CondensedEdgeType, List[str]]:
        """
        Allows access to a mapping which stores the feature keys of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, List[str]]: A mapping which stores the feature keys of each CondensedEdgeTypes
        """
        return {
            condensed_edge_type: list(
                self.condensed_edge_type_to_pos_edge_feature_schema_map[
                    condensed_edge_type
                ].feature_spec.keys()
            )
            for condensed_edge_type in self.condensed_edge_type_to_pos_edge_feature_schema_map
        }

    @property
    def condensed_edge_type_to_hard_neg_edge_feature_dim_map(
        self,
    ) -> Dict[CondensedEdgeType, int]:
        """
        Allows access to a mapping which stores the user-defined negative edge feature
        dimension of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, int]: A mapping which stores the user-defined negative edge feature
            dimension of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_hard_neg_edge_feature_dim_map

    @property
    def condensed_edge_type_to_hard_neg_edge_feature_schema_map(
        self,
    ) -> Dict[CondensedEdgeType, FeatureSchema]:
        """
        Allows access to a mapping which stores the user-defined negative edge feature
        spec, tf schema, and feature index of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, FeatureSchema]: A mapping which stores the user-defined negative edge feature
            spec, tf schema, and feature index of each CondensedEdgeTypes
        """
        return self._condensed_edge_type_to_hard_neg_edge_feature_schema_map

    @property
    def condensed_edge_type_to_hard_neg_edge_feature_keys_map(
        self,
    ) -> Dict[CondensedEdgeType, List[str]]:
        """
        Allows access to a mapping which stores the feature keys of each CondensedEdgeTypes.

        Returns:
            Dict[CondensedEdgeType, List[str]]: A mapping which stores the feature keys of each CondensedEdgeTypes
        """
        return {
            condensed_edge_type: list(
                self.condensed_edge_type_to_hard_neg_edge_feature_schema_map[
                    condensed_edge_type
                ].feature_spec.keys()
            )
            for condensed_edge_type in self.condensed_edge_type_to_hard_neg_edge_feature_schema_map
        }
