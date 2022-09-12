# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random

from cytoolz import curry
import torch


@curry
def set_data(data, n_context, global_rank=-1, world_size=-1):
    examples = load_data(data, global_rank=global_rank, world_size=world_size)
    return Dataset(examples, n_context)


def load_data(fi_data, global_rank=-1, world_size=-1):
    if fi_data.endswith(".jsonl"):
        data = [json.loads(line.strip()) for line in open(fi_data)]
    elif fi_data.endswith(".json"):
        data = json.load(open(fi_data))
    else:
        raise ValueError("data file should be endswith((json, jsonl))")
    examples = []
    for idx, example in enumerate(data):
        # if global_rank > -1 and not idx%world_size==global_rank:
        #     continue
        if not "id" in example:
            example["id"] = idx
        for cix, ctx in enumerate(example["ctxs"]):
            if not "score" in ctx:
                ctx["score"] = 1.0 / (cix + 1)
        examples.append(example)
    return examples


def encode_passages(tokenizer, batch_text_passages, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            truncation=True
        )
        passage_ids.append(p["input_ids"][None])            # To 3D
        passage_masks.append(p["attention_mask"][None])
    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix="question:",
                 title_prefix="title:",
                 passage_prefix="context:"):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()
        self.sep = " [SEP] "

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if "target" in example:
            target = example["target"]
            # return target + self.sep
            return target
        elif "answers" in example:
            # return random.choice(example["answers"]) + self.sep
            return random.choice(example["answers"])
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example["question"]
        target = self.get_target(example)

        if "ctxs" in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example["ctxs"][:self.n_context]
            passages = [f.format(c["title"], c["text"]) for c in contexts]
            scores = [float(c["score"]) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None

        return {
            "index" : index,
            "question" : question,
            "target" : target,
            "passages" : passages,
            "scores" : scores
        }

    def sort_data(self):
        if self.n_context is None or not "score" in self.data[0]["ctxs"][0]:
            return
        for ex in self.data:
            ex["ctxs"].sort(key=lambda x: float(x["score"]), reverse=True)

    def get_example(self, index):
        return self.data[index]


class Collator(object):
    def __init__(self, tokenizer, text_maxlength, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        # batch[0] == Dataset.__getitem__
        assert batch[0]["target"] is not None
        index = torch.tensor([ex["index"] for ex in batch])
        target = [ex["target"] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length = self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding = True,
            return_tensors = "pt",
            truncation = self.answer_maxlength > 0,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example["passages"] is None:
                return [example["question"]]
            return [example["question"] + " " + t for t in example["passages"]]

        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(
            self.tokenizer,
            text_passages,
            self.text_maxlength
        )

        return (index, target_ids, target_mask, passage_ids, passage_masks)


class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        # batch[0] == Dataset.__getitem__
        index = torch.tensor([ex["index"] for ex in batch])

        question = [ex["question"] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            padding = True,
            return_tensors = "pt",
            max_length = self.question_maxlength,
            truncation = True
        )
        question_ids = question["input_ids"]
        question_mask = question["attention_mask"].bool()

        if batch[0]["scores"] is None or batch[0]["passages"] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex["scores"] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex["passages"] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            self.tokenizer,
            passages,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)


class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix="title:",
                 passage_prefix="context:"):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text


class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            padding=True,
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch["input_ids"]
        text_mask = encoded_batch["attention_mask"].bool()

        return index, text_ids, text_mask
