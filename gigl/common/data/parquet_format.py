"""Column and file names of the Parquet files shared by the Parquet writer and reader.

Each node type and each edge type has its own directory under a Parquet prefix. The directory holds files named
``part-XXXXX-of-YYYYY.parquet`` with these columns:

* nodes: ``node_id`` (int64) and, if the node type has features, ``features``.
* edges: ``src`` and ``dst`` (int64) and, if the edge type has features, ``features``.
* ``features`` is a ``FixedSizeList<float32>[feature_dim]``.
* If the entity has packed features, one ``FixedSizeBinary[packed_feature_dim]`` column holds the packed bytes of each
  row. Its name is ``SerializedTFRecordInfo.packed_feature_key``.
* Label columns are optional and named by the label keys.
"""

from typing import Final

NODE_ID_COLUMN: Final[str] = "node_id"
EDGE_SRC_COLUMN: Final[str] = "src"
EDGE_DST_COLUMN: Final[str] = "dst"
FEATURES_COLUMN: Final[str] = "features"
PARQUET_FILE_SUFFIX: Final[str] = ".parquet"
