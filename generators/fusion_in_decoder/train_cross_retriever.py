# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime as dt
import logging
from pathlib import Path
import sys

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import BertTokenizerFast, BertJapaneseTokenizer

from fid.options import Options
from fid.data import RetrieverCollator, set_data
from fid.model import RetrieverConfig, Retriever
from fid.options import Options
from fid.evaluation import eval_batch
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


def train(args, tokenizer, model, optimizer, scheduler, train_dataset, valid_dataset, collator, global_step, best_valid_loss):

    if args.is_main:
        tb_logger = SummaryWriter(Path(args.checkpoint_dir) / args.name)

    train_sampler = DistributedSampler(train_dataset) if args.is_distributed else RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler = train_sampler, 
        batch_size = args.per_gpu_batch_size, 
        drop_last = True, 
        num_workers = 10, 
        collate_fn = collator
    )

    model.train()
    epoch, curr_loss = 0, 0.0
    while global_step < args.total_steps:
        epoch += 1
        if args.is_distributed > 1:
            train_sampler.set_epoch(epoch)
        for idx, batch in enumerate(train_dataloader):
            global_step += 1
            qids, question_ids, question_masks, passage_ids, passage_masks, gold_scores = batch
            question_output, passage_output, score, train_loss = model(
                question_ids = question_ids.cuda(),
                question_mask = question_masks.cuda(),
                passage_ids = passage_ids.cuda(),
                passage_mask = passage_masks.cuda(),
                gold_score = gold_scores.cuda(),
            )

            train_loss.backward()

            if global_step % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            train_loss = util.average_main(train_loss, args)
            curr_loss += train_loss.item()

            if global_step % args.eval_step == 0:
                valid_loss, inversions, avg_topk, idx_topk = evaluate(model, valid_dataset, collator, args)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    if args.is_main:
                        util.save(model, optimizer, scheduler, global_step, best_valid_loss, args, dir_path, 'best_dev')
                model.train()
                if args.is_main:
                    loss = curr_loss / args.eval_step
                    logger.info(f"Evaluation")
                    logger.info(f"  - Steps ... {global_step} / {args.total_steps}")
                    logger.info(f"  - Train Loss ... {loss:.4f}")
                    logger.info(f"  - Valid Loss ... {valid_loss:.4f}")
                    logger.info(f"  - Inversion  ... {inversions:.1f}")
                    # logger.info(f"  - Learning Rate ... {scheduler.get_last_lr()[0]:.6f}"
                    for k in avg_topk:
                        logger.info(f"  - Ave.Top-{k}: {avg_topk[k]:.4f}")
                    for k in idx_topk:
                        logger.info(f"  - Idx.Top-{k}: {idx_topk[k]:.4f}")

                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", valid_loss, global_step)
                        tb_logger.add_scalar("Training", loss, global_step)
                    curr_loss = 0

            if args.is_main and global_step % args.save_freq == 0:
                util.save(model, optimizer, scheduler, global_step, best_valid_loss, args, dir_path, f"step-{global_step}")
            if global_step > args.total_steps:
                break


def evaluate(model, dataset, collator, args):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=args.per_gpu_batch_size,
        drop_last=False, 
        num_workers=10, 
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    eval_loss = []

    avg_topk = {k:[] for k in [1, 2, 5] if k <= args.n_context}
    idx_topk = {k:[] for k in [1, 2, 5] if k <= args.n_context}
    inversions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            idx, question_ids, question_mask, context_ids, context_mask, gold_score = batch

            _, _, scores, loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=context_ids.cuda(),
                passage_mask=context_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            eval_batch(scores, inversions, avg_topk, idx_topk)
            total += question_ids.size(0)

    inversions = util.weighted_average(np.mean(inversions), total, args)[0]
    for k in avg_topk:
        avg_topk[k] = util.weighted_average(np.mean(avg_topk[k]), total, args)[0]
        idx_topk[k] = util.weighted_average(np.mean(idx_topk[k]), total, args)[0]

    return loss, inversions, avg_topk, idx_topk

if __name__ == "__main__":
    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    args = options.parse()

    if args.is_distributed:
        torch.distributed.barrier()
    torch.manual_seed(args.seed)
    fid.slurm.init_distributed_mode(args)
    fid.slurm.init_signal_handler()

    # Tokenizer & Model
    BertClass = BertJapaneseTokenizer if "japanese" in args.model_name_or_path.lower() else BertTokenizerFast
    tokenizer = BertClass.from_pretrained(args.model_name_or_path)
    
    dir_path = Path(args.checkpoint_dir) / args.name
    dir_path.mkdir(parents=True, exist_ok=True)
    if not dir_path.exists() and args.is_main:
        options.print_options(args)

    args.train_batch_size = args.per_gpu_batch_size * max(1, args.world_size)

    global_step = 0
    best_valid_loss = np.inf
    config = RetrieverConfig(
        indexing_dimension = args.indexing_dimension,
        apply_question_mask = not args.no_question_mask,
        apply_passage_mask = not args.no_passage_mask,
        extract_cls = args.extract_cls,
        projection = not args.no_projection,
    )
    if not dir_path.exists() and args.model_path:
        model = Retriever(config, initialize_wBERT=True)
        util.set_dropout(model, args.dropout)
        model = model.to(args.device)
        optimizer, scheduler = util.set_optim(args, model)
    elif dir_path.exists() and args.model_path:
        reset_params = args.model_path is None
        load_path = dir_path / 'checkpoint' / 'latest'
        logger.info(f"Model loaded from {load_path}")
        model, optimizer, scheduler, opt_checkpoint, global_step, best_valid_loss = \
            util.load(Retriever, load_path, args, reset_params=reset_params)
    elif args.model_path is None:
        reset_params = args.model_path is None
        load_path = args.model_path
        logger.info(f"Model loaded from {load_path}")
        model, optimizer, scheduler, opt_checkpoint, global_step, best_valid_loss = \
            util.load(Retriever, load_path, args, reset_params=reset_params)

    model.proj = torch.nn.Linear(768, 256)
    model.norm = torch.nn.LayerNorm(256)
    model.config.indexing_dimension = 256
    model = model.to(args.device)
    optimizer, scheduler = util.set_optim(args, model)

    if args.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.local_rank], 
            output_device=args.local_rank, 
            find_unused_parameters=True,
        )

    # load_data
    collator = RetrieverCollator(
        tokenizer, 
        passage_maxlength = args.passage_maxlength, 
        question_maxlength = args.question_maxlength
    )
    set_data_fn = set_data(global_rank=args.global_rank, world_size=args.world_size)
    train_dataset = set_data_fn(args.train_data, args.n_context)
    valid_dataset = set_data_fn(args.eval_data, args.n_context)

    train(
        args,
        tokenizer, model, optimizer, scheduler,
        train_dataset, valid_dataset, collator,
        global_step, best_valid_loss
    )