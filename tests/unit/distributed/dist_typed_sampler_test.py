"""Unit tests for typed-PPR sampler construction helpers."""

from absl.testing import absltest

from gigl.distributed.utils.dist_typed_sampler import (
    PPRMetaPath,
    build_edge_type_channel_group_edge_type_ids,
    build_typed_ppr_channel_traversal_programs,
    compute_typed_channel_target_counts,
    parse_typed_channel_ratio_groups,
    parse_typed_channel_ratio_specs,
)
from tests.test_assets.distributed.test_dataset import (
    STORY,
    STORY_TO_USER,
    USER,
    USER_TO_STORY,
)
from tests.test_assets.test_case import TestCase

USER_TO_STORY_ALT = (USER, "shares", STORY)
USER_TO_USER_EDGE_1 = (USER, "edge_1", USER)
USER_TO_USER_EDGE_2 = (USER, "edge_2", USER)


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
            compute_typed_channel_target_counts([0.8, 0.1, 0.1], 158),
            [126, 16, 16],
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

    def test_typed_ppr_metapath_channels_parse_and_build_traversal_programs(
        self,
    ) -> None:
        """Verify typed-PPR can use PyG-style ordered metapath channels."""
        node_type_to_edge_types = {
            USER: [USER_TO_STORY, USER_TO_STORY_ALT],
            STORY: [STORY_TO_USER],
        }
        node_types = [USER, STORY]
        edge_type_to_edge_type_id = {
            USER_TO_STORY: 0,
            USER_TO_STORY_ALT: 1,
            STORY_TO_USER: 2,
        }

        typed_channel_specs, typed_channel_ratio_list = parse_typed_channel_ratio_specs(
            {
                PPRMetaPath(
                    path=((USER_TO_STORY, USER_TO_STORY_ALT), STORY_TO_USER),
                    cyclic_from=0,
                ): 1.0,
            }
        )
        assert typed_channel_specs is not None
        assert typed_channel_ratio_list is not None

        self.assertEqual(
            typed_channel_specs,
            [
                PPRMetaPath(
                    path=((USER_TO_STORY, USER_TO_STORY_ALT), (STORY_TO_USER,)),
                    cyclic_from=0,
                )
            ],
        )
        self.assertEqual(typed_channel_ratio_list, [1.0])

        traversal_programs, emitting_state_ids = (
            build_typed_ppr_channel_traversal_programs(
                channel_specs=typed_channel_specs,
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_edge_types=node_type_to_edge_types,
                node_types=node_types,
                edge_dir="out",
            )
        )

        self.assertEqual(
            traversal_programs,
            [
                [
                    [[(0, 1), (1, 1)], []],
                    [[], [(2, 2)]],
                    [[(0, 1), (1, 1)], []],
                ]
            ],
        )
        self.assertEqual(emitting_state_ids, [[2]])

    def test_typed_ppr_metapath_channels_support_prefix_then_repeat(
        self,
    ) -> None:
        """Verify cyclic_from can model edge_1 followed by edge_2 indefinitely."""
        node_type_to_edge_types = {
            USER: [USER_TO_USER_EDGE_1, USER_TO_USER_EDGE_2],
        }
        node_types = [USER]
        edge_type_to_edge_type_id = {
            USER_TO_USER_EDGE_1: 0,
            USER_TO_USER_EDGE_2: 1,
        }

        traversal_programs, emitting_state_ids = (
            build_typed_ppr_channel_traversal_programs(
                channel_specs=[
                    PPRMetaPath(
                        path=(USER_TO_USER_EDGE_1, USER_TO_USER_EDGE_2),
                        cyclic_from=1,
                    )
                ],
                edge_type_to_edge_type_id=edge_type_to_edge_type_id,
                node_type_to_edge_types=node_type_to_edge_types,
                node_types=node_types,
                edge_dir="out",
            )
        )

        self.assertEqual(
            traversal_programs,
            [
                [
                    [[(0, 1)]],
                    [[(1, 1)]],
                ]
            ],
        )
        self.assertEqual(emitting_state_ids, [[1]])

    def test_typed_ppr_metapath_channels_validate_branching_steps(
        self,
    ) -> None:
        """Grouped alternatives must be same-source and same-destination."""
        with self.assertRaisesRegex(ValueError, "grouped-step alternatives"):
            build_typed_ppr_channel_traversal_programs(
                channel_specs=[
                    PPRMetaPath(path=((USER_TO_STORY, USER_TO_USER_EDGE_1),))
                ],
                edge_type_to_edge_type_id={
                    USER_TO_STORY: 0,
                    USER_TO_USER_EDGE_1: 1,
                },
                node_type_to_edge_types={
                    USER: [USER_TO_STORY, USER_TO_USER_EDGE_1],
                    STORY: [],
                },
                node_types=[USER, STORY],
                edge_dir="out",
            )

        with self.assertRaisesRegex(ValueError, "cyclic suffix"):
            build_typed_ppr_channel_traversal_programs(
                channel_specs=[
                    PPRMetaPath(path=(USER_TO_STORY, STORY_TO_USER), cyclic_from=1)
                ],
                edge_type_to_edge_type_id={
                    USER_TO_STORY: 0,
                    STORY_TO_USER: 1,
                },
                node_type_to_edge_types={
                    USER: [USER_TO_STORY],
                    STORY: [STORY_TO_USER],
                },
                node_types=[USER, STORY],
                edge_dir="out",
            )

        with self.assertRaisesRegex(ValueError, "cannot return PPRMetaPath"):
            parse_typed_channel_ratio_groups({PPRMetaPath(path=(USER_TO_STORY,)): 1.0})


if __name__ == "__main__":
    absltest.main()
