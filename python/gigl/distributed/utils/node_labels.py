import torch


def get_labels_from_features(
    feature_and_label_tensor: torch.Tensor, label_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a combined tensor of features and labels, returns the features and labels separately.
    Args:
        feature_and_label_tensor (torch.Tensor): Tensor of features and labels
        label_dim (int): Dimension of the labels
    Returns:
        feature_tensor (torch.Tensor): Tensor of features
        label_tensor (torch.Tensor): Tensor of labels
    """

    _, feature_and_label_dim = feature_and_label_tensor.shape

    feature_dim = feature_and_label_dim - label_dim

    return (
        feature_and_label_tensor[:, :feature_dim],
        feature_and_label_tensor[:, feature_dim:],
    )
