""" Dataset loading script for Jaqket for AIO
"""
import gzip
import json
import logging
from os.path import dirname, join
import sys

from tqdm import tqdm

sys.path.append(join(dirname(__file__), "../../"))
from prepro import RetrievalDataset

logger = logging.getLogger(__name__)


class JaqketAIO(object):
    def __cite__(self):
        return """
            @inproceedings{jaqket-2020,
                author    = {鈴木正敏 and 鈴木潤 and 松田耕史 and 西田京介 and 井之上直也},
                title     = {{JAQKET}: クイズを題材にした日本語 {QA} データセットの構築},
                booktitle = {言語処理学会第26回年次大会},
                year      = {2020},
                pages    = {237--240}
            }
        """
    
    def __project__(self):
        return "https://www.nlp.ecei.tohoku.ac.jp/projects/jaqket/"

    def __init__(self, pathes):
        self.data = self.load_data(pathes)

    def load_data(self, pathes):
        return {
            dtype: RetrievalDataset(path) \
            for dtype, path in pathes.items()
        }
