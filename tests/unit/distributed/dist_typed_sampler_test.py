"""Unit tests for typed-PPR sampler construction helpers."""

from absl.testing import absltest

from gigl.distributed.utils.dist_typed_sampler import (
    build_edge_type_channel_group_edge_type_ids,
    compute_typed_channel_target_counts,
    parse_typed_channel_ratio_groups,
)
from tests.test_assets.distributed.test_dataset import (
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
)
from tests.test_assets.test_case import TestCase


class DistTypedSamplerTest(TestCase):
    def test_typed_ppr_edge_type_channels_parse_and_build_traversal_maps(
        self,
    ) -> None:
        """Verify typed-PPR can use canonical edge-type channels."""
        node_type_to_edge_types = {
            USER: [USER_TO_STORY],
            STORY: [STORY_TO_USER],
        }
        node_types = [USER, STORY]
        edge_type_to_edge_type_id = {
            USER_TO_STORY: 0,
            STORY_TO_USER: 1,
        }

        typed_channel_groups, typed_channel_ratio_list = (
            parse_typed_channel_ratio_groups(
                {
                    USER_TO_STORY: 0.6,
                    (USER_TO_STORY, STORY_TO_USER): 0.4,
                }
            )
        )
        assert typed_channel_groups is not None
        assert typed_channel_ratio_list is not None

        self.assertEqual(
            typed_channel_groups,
            [
                (USER_TO_STORY,),
                (USER_TO_STORY, STORY_TO_USER),
            ],
        )
        self.assertEqual(typed_channel_ratio_list, [0.6, 0.4])
        self.assertEqual(
            compute_typed_channel_target_counts(typed_channel_ratio_list, 7),
            [4, 3],
        )
        self.assertEqual(
            build_edge_type_channel_group_edge_type_ids(
                edge_type_groups=typed_channel_groups,
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_edge_types=node_type_to_edge_types,
                node_types=node_types,
            ),
            [
                [[0], []],
                [[0], [1]],
            ],
        )

        with self.assertRaisesRegex(ValueError, "canonical edge type"):
            parse_typed_channel_ratio_groups({("bad",): 1.0})
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            parse_typed_channel_ratio_groups({USER_TO_STORY: 0.5})
        with self.assertRaisesRegex(ValueError, "non-traversable edge types"):
            build_edge_type_channel_group_edge_type_ids(
                edge_type_groups=[(("unknown", "edge", "type"),)],
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_edge_types=node_type_to_edge_types,
                node_types=node_types,
            )


if __name__ == "__main__":
    absltest.main()
