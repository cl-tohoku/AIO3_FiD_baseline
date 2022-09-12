# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime as dt
import logging
from pathlib import Path
import sys

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter

from transformers import T5Tokenizer, T5ForConditionalGeneration

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


def train(args, tokenizer, model, optimizer, scheduler, train_dataset, valid_dataset, collator, step, best_valid_em, checkpoint_path):

    if args.is_main:
        tb_logger = SummaryWriter(Path(args.checkpoint_dir) / args.name)

    torch.manual_seed(args.global_rank + args.seed)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=collator
    )

    model.train()
    # optimizer.param_groups[0]['capturable'] = True
    epoch, curr_loss, loss = 0, 0.0, -1.0
    with tqdm(desc='[Train]') as pbar:
        while step < args.total_steps:
            epoch += 1
            for i, batch in enumerate(train_dataloader):
                pbar.set_postfix(loss=loss)
                step += 1
                qids, target_ids, target_masks, passage_ids, passage_masks = batch
                outputs = model(
                    input_ids = passage_ids.cuda(),
                    attention_mask = passage_masks.cuda(),
                    labels = target_ids.cuda(),
                )
                
                train_loss = outputs.loss
                train_loss.backward()

                if step % args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                train_loss = util.average_main(train_loss, args)
                curr_loss += train_loss.item()

                if step % args.eval_step == 0:
                    valid_em = evaluate(args, valid_dataset, collator, tokenizer, model)
                    model.train()
                    if args.is_main:
                        if valid_em > best_valid_em:
                            best_valid_em = valid_em
                            util.save(model, optimizer, scheduler, step, best_valid_em, args, checkpoint_path, "best_dev")

                        loss = curr_loss / args.eval_step
                        curr_loss = 0
                        logger.info(f"Evaluation")
                        logger.info(f"  - Steps ... {step} / {args.total_steps}")
                        logger.info(f"  - Train Loss ... {loss:.4f}")
                        logger.info(f"  - Valid EM   ... {valid_em:.4f}")
                        # logger.info(f"lr: {scheduler.get_last_lr()[0]:.5f}")
                        if tb_logger is not None:
                            tb_logger.add_scalar("Evaluation", valid_em, step)
                            tb_logger.add_scalar("Training", loss, step)
                # 学習データのExact matchを評価
                #if step % (50*args.eval_step) == 0:
                #    train_em = evaluate(args, train_dataset, collator, tokenizer, model)
                #    if args.is_main:
                #        logger.info(f"  - Train EM   ... {train_em:.4f}")

                if args.is_main and step % args.save_freq == 0:
                    util.save(model, optimizer, scheduler, step, best_valid_em, args, checkpoint_path, f"step-{step}")
                if step > args.total_steps:
                    break

                pbar.update(1)

        logger.info(f"Training finished")
        logger.info(f"  - total epoch   ... {epoch}")


def evaluate(args, dataset, collator, tokenizer, model):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler = sampler,
        batch_size = args.per_gpu_batch_size,
        drop_last = False,
        num_workers = 10,
        collate_fn = collator
    )
    
    model.eval()
    total = 0
    eval_em = []
    model = model.module if hasattr(model, "module") else model

    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):
            qids, target_ids, target_masks, passage_ids, passage_masks = batch
            outputs = model.generate(
                input_ids = passage_ids.cuda(),
                attention_mask = passage_masks.cuda(),
                max_length = 50
            )

            for bix, output in enumerate(outputs):
                pred = tokenizer.decode(output, skip_special_tokens=True)
                gold = dataset.get_example(qids[bix])["answers"]
                eval_em.append(calc_em(pred, gold))
                total += 1

    eval_em, total = fid.util.weighted_average(np.mean(eval_em), total, args)
    return eval_em



if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    args = options.parse()

    if args.is_distributed:
        torch.distributed.barrier()
    torch.manual_seed(args.seed)
    fid.slurm.init_distributed_mode(args)
    fid.slurm.init_signal_handler()

    # Tokenizer & Model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    checkpoint_path = Path(args.checkpoint_dir) / args.name
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    model_path = args.model_name_or_path or (checkpoint_path / "checkpoint" / "latest")
    # model_path = (checkpoint_path / "checkpoint" / "latest")
    # if Path(model_path).is_dir():
    if Path(model_path).is_file():
        logger.info(f"Model loaded from {model_path}")
        reset_params = args.model_path is not None
        model, optimizer, scheduler, opt_checkpoint, step, best_valid_em = \
            util.load(FiDT5, model_path, args, reset_params=reset_params)
    else:
        t5 = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
        model = FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(args.local_rank)
        optimizer, scheduler = util.set_optim(args, model)
        step, best_valid_em = 0, 0.0

    model.set_checkpoint(args.use_checkpoint)
    if args.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False,
        )

    #load data
    collator = Collator(tokenizer, args.text_maxlength, answer_maxlength=args.answer_maxlength)
    set_data_fn = set_data(global_rank=args.global_rank, world_size=args.world_size)
    train_dataset = set_data_fn(args.train_data, args.n_context)
    valid_dataset = set_data_fn(args.eval_data, args.n_context)

    train(
        args,
        tokenizer, model, optimizer, scheduler,
        train_dataset, valid_dataset, collator,
        step, best_valid_em, checkpoint_path
    )
