import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Generator
import sys

from tqdm import tqdm
from omegaconf import OmegaConf

ROOT_REPOSITORY = Path(__file__).parents[1]
sys.path.append(str(ROOT_REPOSITORY))
import prepro


logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)



class DataFormatter:
    def __init__(self):
        pass

    def keys(self, dataset):
        if isinstance(dataset, prepro.RetrievalDataset):
            return {
                "id": "qid",
                "ctxs": "positive_ctxs",
            }
        else:
            raise NotImplementedError(type(dataset))

    def fusion_in_decoder(self, datasets, dest):
        for dtype, dataset in datasets.items():
            keys = self.keys(dataset)
            with open(f"{dest}/{dtype}.jsonl", "w") as fo:
                for instance in tqdm(dataset, desc=f"[{dtype}]"):
                    if instance["answers"]:
                        obj = {
                            "id": instance[keys["id"]],
                            "question": instance["question"],
                            "target": instance["answers"][0],
                            "answers": instance["answers"],
                            "ctxs": instance["ctxs"]
                        }
                    else:
                        obj = {
                            "id": instance[keys["id"]],
                            "question": instance["question"],
                            "target": "",
                            # "answers": [""],
                            "ctxs": instance["ctxs"]
                        }
                    #if all(o for o in obj.values()):
                    fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                logger.info(f"write ... {fo.name}")



if __name__ == "__main__":
    """ bash
    cd $ROOT_REPOSITORY
    python prepro/convert_dataset.py JaqketAIO fusion_in_decoder
    """

    formatter = DataFormatter()
    FORMAT = list(filter(lambda x: not x.startswith("_"), dir(formatter)))

    parser = argparse.ArgumentParser(description="To create future-aware corpus")
    parser.add_argument("data", type=str, help="key of datasets.yml")
    parser.add_argument("format", choices=FORMAT, help="key of datasets.yml")
    parser.add_argument("-o", "--output_dir", type=str, default="datasets", help="key of datasets.yml")
    args = parser.parse_args()
    
    cfg_file = ROOT_REPOSITORY / "datasets.yml"
    datasets = OmegaConf.load(cfg_file)
    cfg = datasets[args.data]

    dest = Path(args.output_dir) / args.format / args.data
    dest.mkdir(parents=True, exist_ok=True)
    logger.info(f"mkdir ... {dest}")

    module = importlib.import_module(cfg["path"])
    data_class = getattr(module, cfg["class"])
    dataset = data_class(cfg["data"])

    format_fn = getattr(formatter, args.format)
    format_fn(dataset.data, dest)

