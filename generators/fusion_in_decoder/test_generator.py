# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
from datetime import datetime as dt
import logging
from pathlib import Path
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader, SequentialSampler

from transformers import T5Tokenizer

from fid.options import Options
from fid.data import set_data, Collator
from fid.model import FiDT5
from fid.evaluation import calc_em
import fid.slurm
from fid import util


DATETIME = dt.now().strftime("%Y%m%d-%H%M")
FILENAME = __file__.split("/")[-1].rsplit(".", 1)[0]

logging.basicConfig(
    format="%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(filename=f".log_{FILENAME}_{DATETIME}")
    ],
)
logger = logging.getLogger(__name__)


def evaluate(args, dataset, collator, tokenizer, model):
    sampler = SequentialSampler(dataset) 
    dataloader = DataLoader(
        dataset, 
        sampler = sampler, 
        batch_size = args.per_gpu_batch_size,
        num_workers = 20, 
        collate_fn = collator
    )

    model.eval()
    total = 0
    eval_em = []
    model = model.module if hasattr(model, "module") else model

    if args.write_crossattention_scores:
        model.overwrite_forward_crossattention()
        model.reset_score_storage() 
    if args.write_results:
        write_path = Path(args.checkpoint_dir) / args.name / "test_results"
        os.makedirs(write_path, exist_ok=True)
        fw = open(write_path / ("%d.txt" % args.global_rank), "w")

    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):
            if args.write_crossattention_scores:
                model.reset_score_storage()

            qids, target_ids, target_masks, passage_ids, passage_masks = batch
            outputs = model.generate(
                input_ids = passage_ids.cuda(),
                attention_mask = passage_masks.cuda(),
                max_length = 50,
            )

            if args.write_crossattention_scores:
                crossattention_scores = model.get_crossattention_scores(passage_masks.cuda())

            for bix, output in enumerate(outputs):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                example = dataset.data[qids[bix]]
                if "answers" in example:
                    eval_em.append(calc_em(pred, example["answers"]))
                total += 1

                if args.write_results:
                    fw.write(f'{{"qid": "{str(example["id"])}", "prediction": "{pred}"}}\n')
                if args.write_crossattention_scores:
                    for j in range(passage_ids.size(1)):
                        example["ctxs"][j]["score"] = crossattention_scores[bix, j].item()

            if idx % args.eval_print_freq == 0:
                logger.info(f"Process rank: {args.global_rank}, {idx}/{len(dataloader)}")
                if len(eval_em) > 0:
                    logger.info(f"  - Ave.EM = {np.mean(eval_em):.6f}")

    if args.is_distributed:
        torch.distributed.barrier()

    logger.info(f"Process rank: {args.global_rank}, total: {total}")
    if len(eval_em) > 0:
        ave_em = np.mean(eval_em)
        logger.info(f"average = {ave_em:.6f}")
        score, total = util.weighted_average(ave_em, total, args)
        logger.info(f"EM {score:.6f}, Total number of example {total}")
        return score, total
    else:
        return 0.0, total



if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_eval_options()
    args = options.parse()
    
    if args.is_distributed:
        torch.distributed.barrier()
    fid.slurm.init_distributed_mode(args)
    fid.slurm.init_signal_handler()

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model = FiDT5.from_pretrained(args.model_path)
    model = model.to(args.device)

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.world_size)

    # Dataset
    collator = Collator(tokenizer, args.text_maxlength)
    set_data_fn = set_data(global_rank=args.global_rank, world_size=args.world_size)
    eval_dataset = set_data_fn(args.eval_data, args.n_context)
    eval_sampler = SequentialSampler(eval_dataset) 
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler = eval_sampler, 
        batch_size = args.per_gpu_batch_size,
        num_workers = 20, 
        collate_fn = collator
    )

    em, total = evaluate(
        args,
        eval_dataset, collator,
        tokenizer, model
    )

    if args.write_results and args.is_main:
        dir_path = Path(args.checkpoint_dir) / args.name
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / "test_results").mkdir(parents=True, exist_ok=True)
        glob_path = Path(args.checkpoint_dir) / args.name / "test_results"
        write_path = Path(args.checkpoint_dir) / args.name / "final_output.jsonl"
        util.write_output(glob_path, write_path) 
    if args.write_crossattention_scores:
        util.save_distributed_dataset(eval_dataset.data, args)

