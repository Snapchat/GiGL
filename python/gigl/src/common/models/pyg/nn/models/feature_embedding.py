from typing import Dict, Optional

import torch
import torch.nn as nn
from tensorflow_metadata.proto.v0.schema_pb2 import Feature

from gigl.common.logger import Logger
from gigl.common.utils.tensorflow_schema import get_feature_len_from_fixed_len_feature
from gigl.src.common.utils.data.training import filter_features
from gigl.src.data_preprocessor.lib.types import FeatureSchema

logger = Logger()


class FeatureEmbeddingLayer(nn.Module):
    """
    Pass selected features through an embedding layer,
    then concat back with original features set
    """

    def __init__(
        self,
        features_to_embed: Dict[str, int],
        feature_schema: FeatureSchema,
        feature_dim: int,
        aggregation: str = "mean",
        oov_idx: Optional[int] = None,
        padding_idx: Optional[int] = None,
        feature_padding_value_map: Optional[Dict[str, str]] = None,
    ):
        """
        Feature Embedding layer takes in all input features, pass specified features through nn.Embedding layer,
        then return all features transformed (and not transformed).

        Parameters
        ----------
        features_to_embed: feature_name -> feature_dim
        feature_schema: FeatureSchema
        feature_dim: original feature dim
        aggregation: aggregation method after embedding layer, default is "mean"
        oov_idx: idx specifying OOV value, this is used for the special case where a feature uses default
                    tf.compute_and_apply_vocabulary transform and has -1 as OOV value. In this case, we need to
                    add 1 to the whole tensor, so all elements in tensor is >= 0 for nn.Embedding input.
                    If there are multiple OOV buckets specified in tf.compute_and_apply_vocabulary, it will just
                    increase the vocab size by num_oov_buckets, and we will not need to specify oov_idx
        padding_idx: padding_idx for nn.Embedding. Weights and output embedding will be 0 for vocab idx = padding_idx
        feature_padding_value_map: If padding value is provided for a feature,
                                    it will be used to determine the padding_idx based on the vocab by finding the
                                    vocab idx based on the padding value. Integers are represented as strings in vocab.
        """
        super().__init__()

        self.__features_to_embed = features_to_embed
        self.__feature_schema = feature_schema
        self.__feature_embedding_layers = nn.ModuleDict()
        self.__out_dim = feature_dim  # original feature dim
        self.__all_features = list(self.__feature_schema.feature_spec.keys())
        # The non embedding features list should retain the item order of self.__all_features
        self.__non_embed_features = [
            feature
            for feature in self.__all_features
            if feature not in set(features_to_embed.keys())
        ]
        self.__aggregation = aggregation
        assert (
            padding_idx is None or padding_idx >= 0
        ), "padding_idx for embedding layer has to be >= 0"
        self.__padding_idx = padding_idx
        self.__feature_padding_value_map = feature_padding_value_map

        # whether to add 1 to the whole tensor, so all elements in tensor is >= 0 for nn.Embedding input
        # Since tft.compute_and_apply_vocabulary will be 0 based and use -1 as OOV padding
        self.__plus_one = False
        self.__oov_idx: Optional[int] = None
        assert oov_idx is None or oov_idx >= -1, "oov_idx has to be >= -1"
        if oov_idx and oov_idx == -1:
            self.__plus_one = True

        for feature_name, emb_dim in features_to_embed.items():
            feat_dim = get_feature_len_from_fixed_len_feature(
                feature_config=self.__feature_schema.feature_spec[feature_name]
            )
            feat_schema: Feature = self.__feature_schema.schema[feature_name]

            # get vocab size
            # TODO (yliu2-sc) in the future we could get vocab size from somewhere else
            #  instead of schema. If the internal representation moves to TFExample
            #  another option is getting this information from user config, but it's
            #  likely difficult for users to know and provide vocab size as it can change
            #  for certain features
            assert feat_schema.HasField("int_domain"), (
                f"int_domain has to be provided in schema for {feature_name}, "
                f"please check schema.pbtxt"
            )
            assert (
                feat_schema.int_domain.min >= -1
            ), "int_domain.min_value has to be >= -1"
            assert not (
                feat_schema.int_domain.min == -1 and oov_idx != -1
            ), "If int_domain.min_value is -1, oov_idx must also be -1"
            vocab_size = feat_schema.int_domain.max - feat_schema.int_domain.min + 1

            feature_padding_idx: Optional[int]
            if (
                self.__feature_padding_value_map
                and feature_name in self.__feature_padding_value_map
            ):
                feature_padding_value = str(
                    self.__feature_padding_value_map[feature_name]
                )  # type: ignore
                feature_padding_idx = self.__feature_schema.feature_vocab[
                    feature_name
                ].index(feature_padding_value)
            else:
                feature_padding_idx = self.__padding_idx
            if self.__plus_one and feature_padding_idx is not None:
                feature_padding_idx = feature_padding_idx + 1
            self.__feature_embedding_layers[feature_name] = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=emb_dim,
                padding_idx=feature_padding_idx,
            )
            self.__out_dim += emb_dim - feat_dim  # adjust out_dim based on emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_non_embed = filter_features(
            feature_schema=self.__feature_schema,
            feature_names=list(self.__non_embed_features),
            x=x,
        )
        x_emb = [x_non_embed]
        for feature in self.__features_to_embed:
            x_to_emb = filter_features(
                feature_schema=self.__feature_schema, feature_names=[feature], x=x
            ).long()  # embedding layer takes LongTensor
            if self.__plus_one:
                x_to_emb = x_to_emb + 1
            emb_layer = self.__feature_embedding_layers[feature]
            emb = emb_layer(x_to_emb)

            if self.__aggregation == "mean":
                # when taking the mean, we ignore 0 values since they represent padding_idx values
                mask = torch.all(emb != 0, dim=2)  # mask for not all 0 rows
                sum_emb = emb.sum(dim=1)
                count_nonzero = mask.sum(dim=1, keepdim=True).float()
                epsilon = 1e-8
                # for samples that don't have any non-zero values, we set the divisor to epsilon to avoid inf / nan
                safe_divisor = torch.where(
                    torch.isclose(
                        count_nonzero, torch.zeros_like(count_nonzero), atol=epsilon
                    ),
                    torch.full_like(count_nonzero, epsilon),
                    count_nonzero,
                )
                emb_out = sum_emb / safe_divisor
            else:
                # TODO (yliu2-sc) in the future, we can support other aggregation methods, ex. max, concat
                raise NotImplementedError(
                    f"Aggregation method {self.__aggregation} is not supported"
                )
            x_emb.append(emb_out)  # append embedded features

        x_out = torch.cat(x_emb, dim=1)
        return x_out

    @property
    def out_dim(self) -> int:
        """
        Returns the output dimension after Embedding layers are applied
        """
        return self.__out_dim
