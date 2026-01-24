import os
from pathlib import Path

curr_file_parent_dir = Path(__file__).resolve().parent
MAG240_DATASET_PATH = os.path.join(curr_file_parent_dir, "downloads")

NUM_PAPER_FEATURES = 768
TOTAL_NUM_PAPERS = 121751666
TOTAL_NUM_AUTHORS = 122383112
