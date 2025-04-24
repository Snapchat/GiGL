# Task Config Guide

We currently provide the following tasks metadata options:
`NodeBasedTaskMetadata`,`NodeAnchorBasedLinkPredictionTaskMetadata`,`LinkBasedTaskMetadata`. However, only
`NodeAnchorBasedLinkPredictionTaskMetadata` is currently supported.

To Specify the task configuration in GiGL, you will have to specify `TaskMetadata` in your config.

Example of a `NodeAnchorBasedLinkPredictionTaskMetadata` for a graph with two edge types `user-to-story`
`story-to-user`, and where the supervision edge type is `story_to_user`:

```yaml
taskMetadata:
  nodeAnchorBasedLinkPredictionTaskMetadata:
    supervisionEdgeTypes:
      - srcNodeType: user
        relation: to
        dstNodeType: story
```

In this example, the user_to_story edge will be used to sample supervision/positive edges for each user sample.

Example of a `NodeAnchorBasedLinkPredictionTaskMetadata` for a user-user graph where the supervision edge type is
`user_to_user`:

```yaml
  nodeAnchorBasedLinkPredictionTaskMetadata:
    supervisionEdgeTypes:
      - srcNodeType: user
        relation: is_friends_with
        dstNodeType: user
```
