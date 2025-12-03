from torch_geometric.datasets.planetoid import Planetoid

from gigl.common.logger import Logger
from gigl.src.mocking.lib.pyg_datasets_forks import CoraFromGCS

logger = Logger()


def log_stats_for_pyg_planetoid_dataset(dataset: Planetoid):
    logger.info(f"Dataset Info: {dataset}:")
    logger.info("======================")
    logger.info(f"Number of graphs: {len(dataset)}")
    logger.info(f"Number of features: {dataset.num_features}")
    logger.info(f"Number of classes: {dataset.num_classes}")

    logger.info("\n\n======================")
    data = dataset[0]
    logger.info(f"First graph info for dataset: {data}")
    logger.info("======================")
    # Gather some statistics about the graph.
    logger.info(f"Number of nodes: {data.num_nodes}")
    logger.info(f"Number of edges: {data.num_edges}")
    logger.info(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    logger.info(f"Number of training nodes: {data.train_mask.sum()}")
    logger.info(f"Number of validation nodes: {data.val_mask.sum()}")
    logger.info(f"Number of testing nodes: {data.test_mask.sum()}")

    logger.info(
        f"""Training node label rate: {int(
            data.train_mask.sum() + data.test_mask.sum() + data.val_mask.sum()
        ) / data.num_nodes:.2f}"""
    )
    logger.info(f"Has isolated nodes: {data.has_isolated_nodes()}")
    logger.info(f"Has self-loops: {data.has_self_loops()}")
    logger.info(f"Is undirected: {data.is_undirected()}")


def get_pyg_cora_dataset(store_at: str = "/tmp/Cora") -> Planetoid:
    """Cora graph is the graph in the first index in the returned dataset
    i.e. the Planetoid object is subscriptable, data = dataset[0]
    Train and tests masks are defined by `train_mask` and `test_mask`` properties on data.


    Returns:
        torch_geometric.datasets.planetoid.Planetoid
    """
    # Fetch the dataset
    dataset = CoraFromGCS(root=store_at, name="Cora")
    return dataset
