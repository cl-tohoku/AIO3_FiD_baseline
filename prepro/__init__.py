import gzip
import json
import logging

from tqdm import tqdm
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class RetrievalDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> dict:
        return self.data[idx]

    def load_data(self, path):
        open_fn = gzip.open if path.endswith(".gz") else open
        logger.info(f"loading ... {path}")
        return json.load(open_fn(path, "rt"))
