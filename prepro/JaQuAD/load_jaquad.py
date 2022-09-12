import logging
from datasets import load_dataset


logger = logging.getLogger(__name__)


class JaQuAD:
    def __init__(self):
        logger.info("\033[32m" + "LOAD: JaQuAD" + "\033[0m")
        self.data = load_dataset("SkelterLabsInc/JaQuAD")

    def __doc__(self):
        return "https://huggingface.co/datasets/SkelterLabsInc/JaQuAD"

    def __arxiv__(self):
        return "https://arxiv.org/abs/2202.01764"



if __name__ == "__main__":
    dataset = JaQuAD()
