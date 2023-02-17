# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning models for NER and POS tagging."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

from cvxopt import matrix, solvers
# from tkinter import _flatten

import numpy as np
import torch
from seqeval.metrics import precision_score, recall_score, f1_score
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from panx_model import CustomXLMRoBertaForTokenClassification, CustomXLMRobertaConfig
from utils.utils_tag_remove import convert_examples_to_features
from utils.utils_tag_remove import get_labels
from utils.utils_tag_remove import read_examples_from_file

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    WEIGHTS_NAME,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    XLMConfig,
    XLMTokenizer,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForTokenClassification
)
from xlm import XLMForTokenClassification

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys())
     for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
    ()
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
    "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
    "xlmr-mh": (CustomXLMRobertaConfig, (CustomXLMRoBertaForTokenClassification), XLMRobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def merge_dataset(dataset_list, return_task_batch_list=True):
    iterators = [iter(d) for d in dataset_list]
    n = len(dataset_list)
    while True:
        batch_list = []
        for i in range(n):
            try:
                batch = next(iterators[i])
            except StopIteration:
                iterators[i] = iter(dataset_list[i])
                batch = next(iterators[i])
            batch_list.append(batch)
            if not return_task_batch_list:
                yield batch
        if return_task_batch_list:
            yield batch_list


def acc_less_forgetting_direction(grad_list, last_opt_grad):
    grad_list = [[torch.flatten(grad_list[t][i]) for i in range(len(grad_list[0]))] for t in range(len(grad_list))]
    last_opt_grad = [torch.flatten(last_opt_grad[i]) for i in range(len(last_opt_grad))]
    interval_layers = 10
    solvers.options['show_progress'] = False
    w = torch.ones(len(grad_list)) / len(grad_list)
    blocks_P = []
    blocks_q = []
    for layer in range(len(grad_list[0]) // interval_layers + 1):
        start = layer * interval_layers
        end = layer * interval_layers + interval_layers
        final_end = layer * interval_layers + len(grad_list[0]) % interval_layers

        if layer == len(grad_list[0]) // interval_layers:
            block = [torch.cat(grad_list[t][start:final_end]) for t in range(len(grad_list))]
            block = torch.stack(block)
            last_grad_block = torch.cat(last_opt_grad[start:final_end], dim=0)
            last_grad_block = torch.reshape(last_grad_block, (last_grad_block.size()[0], 1))
        else:
            block = [torch.cat(grad_list[t][start:end], dim=0) for t in range(len(grad_list))]
            block = torch.stack(block)
            last_grad_block = torch.cat(last_opt_grad[start:end], dim=0)
            last_grad_block = torch.reshape(last_grad_block, (last_grad_block.size()[0], 1))

        block_P = block.mm(block.t())
        blocks_P.append(block_P)
        # last_grad_block = torch.reshape(torch.flatten(last_opt_grad[layer]), (torch.flatten(last_opt_grad[layer]).size()[0], 1))
        block_q = block.mm(last_grad_block)
        blocks_q.append(block_q)
    P_ = sum(blocks_P)
    q_ = sum(blocks_q).t()

    P = matrix(P_.tolist())
    q = 2 * matrix(sum(q_.tolist(), []))
    I = np.identity(len(grad_list))
    G_ = P_.tolist() + I.tolist()
    G_ = list(map(list, zip(*G_)))
    G = -1 * matrix(G_)
    h = matrix(np.zeros(2 * len(grad_list)))
    A = matrix(np.ones(len(grad_list)).reshape(1, -1))
    b = matrix(1.)
    ans = solvers.qp(P, q, G, h, A, b)

    for i in range(len(grad_list)):
        w[i] = ans['x'][i]
    # w = torch.ones(len(grad_list)) / len(grad_list)
    return w


def open_set_less_forgetting_direction(grad_list, last_opt_grad):
    solvers.options['show_progress'] = False
    w = torch.ones(len(grad_list)) / len(grad_list)
    blocks_P = []
    blocks_q = []
    for layer in range(len(grad_list[0])):
        block = torch.stack([torch.flatten(grad_list[t][layer]) for t in range(len(grad_list))])
        block_P = block.mm(block.t())
        blocks_P.append(block_P)
        last_grad_block = torch.reshape(torch.flatten(last_opt_grad[layer]),
                                        (torch.flatten(last_opt_grad[layer]).size()[0], 1))
        block_q = block.mm(last_grad_block)
        blocks_q.append(block_q)
    P_ = sum(blocks_P)
    q_ = sum(blocks_q).t()

    P = matrix(P_.tolist())
    q = 2 * matrix(sum(q_.tolist(), []))
    I = np.identity(len(grad_list))
    G = -1 * matrix(P_.tolist())
    h = matrix(np.zeros(len(grad_list)))
    A = matrix(np.ones(len(grad_list)).reshape(1, -1))
    b = matrix(1.)
    ans = solvers.qp(P, q, G, h, A, b)

    for i in range(len(grad_list)):
        w[i] = ans['x'][i]
    # print(w)
    # w = torch.ones(len(grad_list)) / len(grad_list)
    return w


def tradeoff_less_forgetting_direction(grad_list, last_opt_grad):
    # solvers.options['feastol'] = 0.001
    alpha = -0.00001
    solvers.options['show_progress'] = False
    w = torch.ones(len(grad_list)) / len(grad_list)
    blocks_P = []
    blocks_q = []
    for layer in range(len(grad_list[0])):
        block = torch.stack([torch.flatten(grad_list[t][layer]) for t in range(len(grad_list))])
        block_P = block.mm(block.t())
        blocks_P.append(block_P)
        last_grad_block = torch.reshape(torch.flatten(last_opt_grad[layer]),
                                        (torch.flatten(last_opt_grad[layer]).size()[0], 1))
        block_q = block.mm(last_grad_block)
        blocks_q.append(block_q)
    P_ = sum(blocks_P)
    q_ = sum(blocks_q).t()

    P = matrix(P_.tolist())
    q = 2 * matrix(sum(q_.tolist(), []))
    I = np.identity(len(grad_list))
    G_ = P_.tolist() + I.tolist()
    G_ = list(map(list, zip(*G_)))
    G = -1 * matrix(G_)
    h = alpha * matrix(np.concatenate((np.ones(len(grad_list)), np.zeros(len(grad_list)))))
    A = matrix(np.ones(len(grad_list)).reshape(1, -1))
    b = matrix(1.)
    ans = solvers.qp(P, q, G, h, A, b)

    for i in range(len(grad_list)):
        w[i] = ans['x'][i]
    # w = torch.ones(len(grad_list)) / len(grad_list)
    return w


def save_mem_less_forgetting_direction(grad_list, last_opt_grad):
    # alpha = -0.00001
    # lamda = 1.2
    solvers.options['show_progress'] = False
    num_task = len(grad_list)
    number_layers = len(grad_list[0])
    w = torch.ones(num_task) / num_task
    blocks_P = []
    blocks_q = []
    for layer in range(number_layers):
        block_P = torch.zeros(num_task, num_task)
        block_q = torch.zeros(num_task, 1)
        for i in range(num_task):
            for j in range(num_task):
                block_P[i][j] = torch.sum(torch.flatten(grad_list[i][layer]) * torch.flatten(grad_list[j][layer]))
        for i in range(num_task):
            block_q[i] = torch.sum(torch.flatten(grad_list[i][layer]) * torch.flatten(last_opt_grad[layer]))
        blocks_P.append(block_P)
        blocks_q.append(block_q)
    P_ = sum(blocks_P)
    q_ = sum(blocks_q).t()

    P = matrix(P_.tolist())
    q = 2 * matrix(sum(q_.tolist(), []))
    I = np.identity(len(grad_list))
    G_ = P_.tolist() + I.tolist()
    G_ = list(map(list, zip(*G_)))
    G = -1 * matrix(G_)
    h = matrix(np.zeros(2 * len(grad_list)))
    # h = alpha * matrix(np.concatenate((np.ones(len(grad_list)), np.zeros(len(grad_list)))))
    A = matrix(np.ones(len(grad_list)).reshape(1, -1))
    b = matrix(1.)
    ans = solvers.qp(P, q, G, h, A, b)

    for i in range(len(grad_list)):
        w[i] = ans['x'][i]
    # w = torch.ones(len(grad_list)) / len(grad_list)
    return w


def less_forgetting_direction(grad_list, last_opt_grad):
    solvers.options['show_progress'] = False
    w = torch.ones(len(grad_list)) / len(grad_list)
    blocks_P = []
    blocks_q = []
    for layer in range(len(grad_list[0])):
        b = []
        for t in range(len(grad_list)):
            b.append(torch.flatten(grad_list[t][layer]))
        block = torch.stack(b)
        # block = torch.stack([torch.flatten(grad_list[t][layer]) for t in range(len(grad_list))])
        block_P = block.mm(block.t())
        blocks_P.append(block_P)
        last_grad_block = torch.reshape(torch.flatten(last_opt_grad[layer]),
                                        (torch.flatten(last_opt_grad[layer]).size()[0], 1))
        block_q = block.mm(last_grad_block)
        blocks_q.append(block_q)
    P_ = sum(blocks_P)
    q_ = sum(blocks_q).t()

    P = matrix(P_.tolist())
    q = 2 * matrix(sum(q_.tolist(), []))
    I = np.identity(len(grad_list))
    G_ = P_.tolist() + I.tolist()
    G_ = list(map(list, zip(*G_)))
    G = -1 * matrix(G_)
    h = matrix(np.zeros(2 * len(grad_list)))
    A = matrix(np.ones(len(grad_list)).reshape(1, -1))
    b = matrix(1.)
    ans = solvers.qp(P, q, G, h, A, b)

    for i in range(len(grad_list)):
        w[i] = ans['x'][i]
    # w = torch.ones(len(grad_list)) / len(grad_list)
    return w


def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id, langs, lang2id=None):
    """Train the model."""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_dataloaders = []
    upd_per_epoch = 0
    num_train_samples = 0
    for t in range(len(train_dataset)):
        train_sampler = RandomSampler(train_dataset[t]) if args.local_rank == -1 else DistributedSampler(
            train_dataset[t])
        task_dataloader = DataLoader(train_dataset[t], sampler=train_sampler, batch_size=args.train_batch_size)
        if len(task_dataloader) > upd_per_epoch:
            upd_per_epoch = len(task_dataloader)
        train_dataloaders.append(task_dataloader)
        num_train_samples += len(task_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (upd_per_epoch // args.gradient_accumulation_steps) + 1
    else:
        t_total = upd_per_epoch // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", num_train_samples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = 0.0
    best_checkpoint = None
    patience = 0
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    grad_cache = []
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    grad_opt_last = None
    set_seed(args)  # Add here for reproductibility (even between python 2 and 3)

    w = torch.ones(len(langs)) / len(langs)
    w = w.to(args.device)

    last_opt_grad = []
    id2lang = {v: k for k, v in lang2id.items()}
    for _ in train_iterator:
        pbar = tqdm(total=upd_per_epoch, desc="Iteration")
        # epoch_iterator = tqdm(merge_dataset(train_dataloader), desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, all_task_batch in enumerate(merge_dataset(train_dataloaders)):
            pbar.update(1)
            if step > upd_per_epoch - 1:
                pbar.close()
                break
            loss_t = {}
            grad_list = []

            for t, batch in enumerate(all_task_batch):
                # task = list(lang2id.keys())[list(lang2id.values()).index(batch[4][0])]  # id2lang
                task = id2lang[batch[4][0].item()]
                model.train()
                batch = tuple(t.to(args.device) for t in batch if t is not None)
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]}

                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

                if args.model_type in ["xlm"]:
                    inputs["langs"] = batch[4]

                if args.model_type in ["xlmr-mh"]:
                    inputs["task"] = task

                outputs = model(**inputs)
                loss_t[task] = outputs[0]

                if args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel training
                    loss_t[task] = loss_t[task].mean()
                if args.gradient_accumulation_steps > 1:
                    loss_t[task] = loss_t[task] / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss_t[task], optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss_t[task].backward(retain_graph=True)
                    grad_list.append([p.grad.clone().detach() for p in model.parameters() if p.grad is not None])
                    model.zero_grad()

                tr_loss += loss_t[task].item()
            if step > 0:
                w = save_mem_less_forgetting_direction(grad_list, last_opt_grad)

            opt_grad = []
            for layer, p in enumerate([p for p in model.parameters() if p.grad is not None]):
                for t in range(len(grad_list)):
                    if p.grad is not None:
                        p.grad += w[t] * grad_list[t][layer]
                opt_grad.append(p.grad.clone().detach())
            last_opt_grad = opt_grad

            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.local_rank == -1 and args.evaluate_during_training:
                    # Only evaluate on single GPU otherwise metrics may not average well
                    results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                          lang=args.train_langs, lang2id=lang2id)
                    for key, value in results.items():
                        tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                if args.save_only_best_checkpoint:
                    result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                         prefix=global_step, lang=args.train_langs, lang2id=lang2id)
                    if result["f1"] > best_score:
                        logger.info("result['f1']={} > best_score={}".format(result["f1"], best_score))
                        best_score = result["f1"]
                        # Save the best model checkpoint
                        output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        best_checkpoint = output_dir
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        # Take care of distributed/parallel training
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving the best model checkpoint to %s", output_dir)
                        logger.info("Reset patience to 0")
                        patience = 0
                    else:
                        patience += 1
                        logger.info("Hit patience={}".format(patience))
                        if args.eval_patience > 0 and patience > args.eval_patience:
                            logger.info("early stop! patience={}".format(patience))
                            train_iterator.close()
                            if args.local_rank in [-1, 0]:
                                tb_writer.close()
                            return global_step, tr_loss / global_step
                else:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix="", lang="en", lang2id=None,
             print_result=True):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, lang=lang,
                                           lang2id=lang2id)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_dataloader = []
    eval_per_epoch = 0
    num_eval_samples = 0
    for t in range(len(eval_dataset)):
        task_eval_sampler = SequentialSampler(eval_dataset[t]) if args.local_rank == -1 else DistributedSampler(
            eval_dataset[t])
        task_eval_dataloader = DataLoader(eval_dataset[t], sampler=task_eval_sampler, batch_size=args.eval_batch_size)
        eval_dataloader.append(task_eval_dataloader)
        if len(task_eval_dataloader) > eval_per_epoch:
            eval_per_epoch = len(task_eval_dataloader)
        num_eval_samples += len(task_eval_dataloader)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s in %s *****" % (prefix, lang))
    logger.info("  Num examples = %d", num_eval_samples)
    logger.info("  Batch size = %d", args.eval_batch_size)

    langs = lang.split(',')
    eval_loss = {}
    preds = {}
    out_label_ids = {}
    nb_eval_steps = {}
    for l in langs:
        eval_loss[l] = 0.0
        preds[l] = None
        out_label_ids[l] = None
        nb_eval_steps[l] = 0

    id2lang = {v: k for k, v in lang2id.items()}
    model.eval()
    evalbar = tqdm(total=eval_per_epoch, desc="Evaluating")
    for step, all_eval_batch in enumerate(merge_dataset(eval_dataloader)):
        evalbar.update(1)
        if step > eval_per_epoch - 1:
            evalbar.close()
            break

        tmp_eval_loss = {}
        logits = {}
        for t, batch in enumerate(all_eval_batch):
            # task = list(lang2id.keys())[list(lang2id.values()).index(batch[4][0])]
            task = id2lang[batch[4][0].item()]
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],
                          "attention_mask": batch[1],
                          "labels": batch[3]}

                if args.model_type != "distilbert":
                    # XLM and RoBERTa don"t use segment_ids
                    inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None

                if args.model_type == 'xlm':
                    inputs["langs"] = batch[4]

                if args.model_type in ["xlmr-mh"]:
                    inputs["task"] = task

                outputs = model(**inputs)
                tmp_eval_loss[task], logits[task] = outputs[:2]

                if args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel evaluating
                    tmp_eval_loss[task] = tmp_eval_loss[task].mean()

                eval_loss[task] += tmp_eval_loss[task].item()

            nb_eval_steps[task] += 1
            if preds[task] is None:
                preds[task] = logits[task].detach().cpu().numpy()
                out_label_ids[task] = inputs["labels"].detach().cpu().numpy()
            else:
                preds[task] = np.append(preds[task], logits[task].detach().cpu().numpy(), axis=0)
                out_label_ids[task] = np.append(out_label_ids[task], inputs["labels"].detach().cpu().numpy(), axis=0)

    results = {}
    loss_avg = 0
    precision_avg = 0
    recall_avg = 0
    f1_avg = 0
    for task in langs:
        if nb_eval_steps[task] == 0:
            results[task] = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
            results_avg = {k: 0 for k in ["loss", "precision", "recall", "f1"]}
        else:
            eval_loss[task] = eval_loss[task] / nb_eval_steps[task]
            preds[task] = np.argmax(preds[task], axis=2)

            label_map = {i: label for i, label in enumerate(labels)}

            out_label_list = [[] for _ in range(out_label_ids[task].shape[0])]
            preds_list = [[] for _ in range(out_label_ids[task].shape[0])]

            for i in range(out_label_ids[task].shape[0]):
                for j in range(out_label_ids[task].shape[1]):
                    if out_label_ids[task][i, j] != pad_token_label_id:
                        out_label_list[i].append(label_map[out_label_ids[task][i][j]])
                        preds_list[i].append(label_map[preds[task][i][j]])

            results[task] = {
                "loss": eval_loss[task],
                "precision": precision_score(out_label_list, preds_list),
                "recall": recall_score(out_label_list, preds_list),
                "f1": f1_score(out_label_list, preds_list)
            }

            loss_avg += results[task]["loss"] / len(langs)
            precision_avg += results[task]["precision"] / len(langs)
            recall_avg += results[task]["recall"] / len(langs)
            f1_avg += results[task]["f1"] / len(langs)

        if print_result:
            logger.info("***** Evaluation result %s in %s *****" % (prefix, task))
            for key in sorted(results[task].keys()):
                logger.info("  %s = %s", key, str(results[task][key]))

    results_avg = {
        "loss": loss_avg,
        "precision": precision_avg,
        "recall": recall_avg,
        "f1": f1_avg
    }
    logger.info("***** Evaluation result %s in %s *****" % (prefix, "average"))
    for key in sorted(results_avg.keys()):
        logger.info("  %s = %s", key, str(results_avg[key]))

    return results_avg, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode, lang, lang2id=None, few_shot=-1):
    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}_{}".format(mode, lang,
                                                                                   list(filter(None,
                                                                                               args.model_name_or_path.split(
                                                                                                   "/"))).pop(),
                                                                                   str(args.max_seq_length)))
    dataset = []
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        langs = lang.split(',')
        logger.info("all languages = {}".format(lang))
        features = []
        for lg in langs:
            data_file = os.path.join(args.data_dir, lg, "{}.{}".format(mode, args.model_name_or_path))
            logger.info("Creating features from dataset file at {} in language {}".format(data_file, lg))
            examples = read_examples_from_file(data_file, lg, lang2id)
            features_lg = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                       cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                       cls_token=tokenizer.cls_token,
                                                       cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                       sep_token=tokenizer.sep_token,
                                                       sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]),
                                                       pad_on_left=bool(args.model_type in ["xlnet"]),
                                                       pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[
                                                           0],
                                                       pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                       pad_token_label_id=pad_token_label_id,
                                                       lang=lg,
                                                       remove_special_token=lg in ["zh", "ja", "th"]
                                                       )
            features.extend(features_lg)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file {}, len(features)={}".format(cached_features_file, len(features)))
            torch.save(features, cached_features_file)

    # Make sure only the first process in distributed training process
    # the dataset, and the others will use the cache
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if few_shot > 0 and mode == 'train':
        logger.info("Original no. of examples = {}".format(len(features)))
        features = features[: few_shot]
        logger.info('Using few-shot learning on {} examples'.format(len(features)))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)

    tasks = lang.split(',')
    for i, task in enumerate(tasks):
        langid = lang2id.get(task)
        data_belong_task = (all_langs == langid).nonzero().reshape(-1)
        if len(data_belong_task) > 0:
            input_ids_task = torch.index_select(all_input_ids, 0, data_belong_task)
            attention_mask_task = torch.index_select(all_input_mask, 0, data_belong_task)
            segment_ids_task = torch.index_select(all_segment_ids, 0, data_belong_task)
            labels_task = torch.index_select(all_label_ids, 0, data_belong_task)
            langs_task = torch.index_select(all_langs, 0, data_belong_task)
        dataset.append(TensorDataset(input_ids_task, attention_mask_task, segment_ids_task, labels_task, langs_task))
    # if args.model_type == 'xlm' and features[0].langs is not None:
    #   all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
    #   logger.info('all_langs[0] = {}'.format(all_langs[0]))
    #   dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    # else:
    #   dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the training files for the NER/POS task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--labels", default="", type=str,
                        help="Path to a file containing all labels. If not specified, NER/POS labels are used.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")
    parser.add_argument("--do_predict_dev", action="store_true",
                        help="Whether to run predictions on the dev set.")
    parser.add_argument("--init_checkpoint", default=None, type=str,
                        help="initial checkpoint for train/predict")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--few_shot", default=-1, type=int,
                        help="num of few-shot exampes")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_only_best_checkpoint", action="store_true",
                        help="Save only the best checkpoint during training")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--predict_langs", type=str, default="en", help="prediction languages")
    parser.add_argument("--train_langs", default="en", type=str,
                        help="The languages in the training sets.")
    parser.add_argument("--log_file", type=str, default=None, help="log file")
    parser.add_argument("--eval_patience", type=int, default=-1,
                        help="wait N times of decreasing dev score before early stop during training")
    args = parser.parse_args()

    print("="*100)
    print(args.model_type)

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which sychronizes nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(handlers=[logging.FileHandler(args.log_file), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info("Input args: %r" % args)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare NER/POS task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.init_checkpoint:
        logger.info("loading from init_checkpoint={}".format(args.init_checkpoint))
        model = model_class.from_pretrained(args.init_checkpoint,
                                            config=config,
                                            cache_dir=args.init_checkpoint)
    else:
        logger.info("loading from cached model = {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool(".ckpt" in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,)
    # lang2id = config.lang2id if args.model_type == "xlm" else None
    #   lang2id = {'af': 0, 'als': 1, 'am': 2, 'an': 3, 'ang': 4, 'ar': 5, 'arz': 6, 'ast': 7, 'az': 8, 'bar': 9, 'be': 10, 'bg': 11, 'bn': 12, 'br': 13, 'bs': 14, 'ca': 15, 'ceb': 16, 'ckb': 17, 'cs': 18, 'cy': 19, 'da': 20, 'de': 21, 'el': 22, 'en': 23, 'eo': 24, 'es': 25, 'et': 26, 'eu': 27, 'fa': 28, 'fi': 29, 'fr': 30, 'fy': 31, 'ga': 32, 'gan': 33, 'gl': 34, 'gu': 35, 'he': 36, 'hi': 37, 'hr': 38, 'hu': 39, 'hy': 40, 'ia': 41, 'id': 42, 'is': 43, 'it': 44, 'ja': 45, 'jv': 46, 'ka': 47, 'kk': 48, 'kn': 49, 'ko': 50, 'ku': 51, 'la': 52, 'lb': 53, 'lt': 54, 'lv': 55, 'mk': 56, 'ml': 57, 'mn': 58, 'mr': 59, 'ms': 60, 'my': 61, 'nds': 62, 'ne': 63, 'nl': 64, 'nn': 65, 'no': 66, 'oc': 67, 'pl': 68, 'pt': 69, 'ro': 70, 'ru': 71, 'scn': 72, 'sco': 73, 'sh': 74, 'si': 75, 'simple': 76, 'sk': 77, 'sl': 78, 'sq': 79, 'sr': 80, 'sv': 81, 'sw': 82, 'ta': 83, 'te': 84, 'th': 85, 'tl': 86, 'tr': 87, 'tt': 88, 'uk': 89, 'ur': 90, 'uz': 91, 'vi': 92, 'war': 93, 'wuu': 94, 'yi': 95, 'zh': 96, 'zh_classical': 97, 'zh_min_nan': 98, 'zh_yue': 99, 'yo': 100, 'qu': 101, 'pa': 102}
    lang2id = {'ar': 0, 'he': 1, 'vi': 2, 'id': 3, 'jv': 4, 'ms': 5, 'tl': 6, 'eu': 7, 'ml': 8, 'ta': 9, 'te': 10,
               'af': 11, 'nl': 12, 'en': 13, 'de': 14, 'el': 15, 'bn': 16, 'hi': 17, 'mr': 18, 'ur': 19, 'fa': 20,
               'fr': 21, 'it': 22, 'pt': 23, 'es': 24, 'bg': 25, 'ru': 26, 'ja': 27, 'ka': 28, 'ko': 29, 'th': 30,
               'sw': 31, 'yo': 32, 'my': 33, 'zh': 34, 'kk': 35, 'tr': 36, 'et': 37, 'fi': 38, 'hu': 39, 'qu': 40,
               'pl': 41, 'uk': 42, 'az': 43, 'lt': 44, 'pa': 45, 'gu': 46, 'ro': 47}
    logger.info("Using lang2id = {}".format(lang2id))

    # Make sure only the first process in distributed training loads model/vocab
    if args.local_rank == 0:
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train",
                                                lang=args.train_langs, lang2id=lang2id, few_shot=args.few_shot)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id,
                                     args.train_langs, lang2id)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use default names for the model,
    # you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        # Save model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        logger.info("Saving model checkpoint to %s", args.output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Initialization for evaluation
    results = {}
    if args.init_checkpoint:
        best_checkpoint = args.init_checkpoint
    elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
        best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
    else:
        best_checkpoint = args.output_dir
    best_f1 = 0

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint).to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step,
                                 lang=args.train_langs, lang2id=lang2id)
            if result["f1"] > best_f1:
                best_checkpoint = checkpoint
                best_f1 = result["f1"]
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))
            writer.write("best checkpoint = {}, best f1 = {}\n".format(best_checkpoint, best_f1))

    # Prediction
    if args.do_predict and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "a") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'test.{}'.format(args.model_name_or_path))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "test_{}_predictions.txt".format(lang))
                infile = os.path.join(args.data_dir, lang, "test.{}".format(args.model_name_or_path))
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)

    # Predict dev set
    if args.do_predict_dev and args.local_rank in [-1, 0]:
        logger.info("Loading the best checkpoint from {}\n".format(best_checkpoint))
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(best_checkpoint)
        model.to(args.device)

        output_test_results_file = os.path.join(args.output_dir, "dev_results.txt")
        with open(output_test_results_file, "w") as result_writer:
            for lang in args.predict_langs.split(','):
                if not os.path.exists(os.path.join(args.data_dir, lang, 'dev.{}'.format(args.model_name_or_path))):
                    logger.info("Language {} does not exist".format(lang))
                    continue
                result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev",
                                               lang=lang, lang2id=lang2id)

                # Save results
                result_writer.write("=====================\nlanguage={}\n".format(lang))
                for key in sorted(result.keys()):
                    result_writer.write("{} = {}\n".format(key, str(result[key])))
                # Save predictions
                output_test_predictions_file = os.path.join(args.output_dir, "dev_{}_predictions.txt".format(lang))
                infile = os.path.join(args.data_dir, lang, "dev.{}".format(args.model_name_or_path))
                idxfile = infile + '.idx'
                save_predictions(args, predictions, output_test_predictions_file, infile, idxfile)


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
    # Save predictions
    with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
        text = text_reader.readlines()
        index = idx_reader.readlines()
        assert len(text) == len(index)

    # Sanity check on the predictions
    with open(output_file, "w") as writer:
        example_id = 0
        prev_id = int(index[0])
        for line, idx in zip(text, index):
            if line == "" or line == "\n":
                example_id += 1
            else:
                cur_id = int(idx)
                output_line = '\n' if cur_id != prev_id else ''
                if output_word_prediction:
                    output_line += line.split()[0] + '\t'
                output_line += predictions[example_id].pop(0) + '\n'
                writer.write(output_line)
                prev_id = cur_id


if __name__ == "__main__":
    main()
