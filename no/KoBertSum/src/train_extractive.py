#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import copy

# import pandas as pd
import torch

import distributed
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']


def train_multi_ext(args):
    """ Spawns 1 process per GPU """
    init_logger()

    nb_gpu = args.world_size
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_ext(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)



def validate_ext(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_ext(args, device_id, cp, step)
    else:
        while (True):
            # cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files = glob.glob(os.path.join(args.model_path, 'model_step_15000.pt'))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_ext(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)

def train_validate_ext(step, args, device_id):
    timestep = 0
    # cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    checkpoint_path = os.path.join(args.model_path, 'model_step_%d.pt' % step)
    cp_files = glob.glob(os.path.join(args.model_path, 'model_step_500.pt'))
    cp_files.sort(key=os.path.getmtime)
    if (cp_files):
        cp = cp_files[-1]
        time_of_cp = os.path.getmtime(cp)
        if (not os.path.getsize(cp) > 0):
            print("sleeping 60 sec...")
            time.sleep(60)
        if (time_of_cp > timestep):
            timestep = time_of_cp
            step = int(cp.split('.')[-2].split('_')[-1])
            validate(args, device_id, cp, step)
            test_ext(args, device_id, cp, step)

    cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    if (cp_files):
        cp = cp_files[-1]
        time_of_cp = os.path.getmtime(cp)
        # if (time_of_cp > timestep):
        #     continue
    else:
        print("sleeping 300 sec...")
        time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else f"cuda:{args.visible_gpus}"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    # test_from = '/tmp/pycharm_project_138/no/KoBertSum/ext/models/1209_1236/model_step_8000.pt'
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    # print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    # valid_iter_2 = copy.deepcopy(valid_iter)
    stats, predicted_labels, true_labels = trainer.validate(valid_iter, step)
    #print("predicted:", predicted_labels)
    # print("true:", true_labels)
    #predicted_labels, true_labels = trainer.test_for_results_only(valid_iter_2, step)
    results = []
    for i in range(len(true_labels)):
        x = 0
        for j in range(len(predicted_labels[i])):
            if predicted_labels[i][j] in true_labels[i]:
                x += 1
        results.append(x)
    for i in range(4):
        print(i, "hits:", results.count(i))
    print("Hit rate", (results.count(1) + results.count(2) * 2 + results.count(3) * 3)/((results.count(0) + results.count(1) + results.count(2) + results.count(3)) * 3))
    # ext_df = pd.DataFrame({"predict": predicted_labels, "answer": true_labels, "hit_count": results})
    # ext_df.to_csv(f"validation_hit_stats_step_{step}")
    return stats.xent()


def test_ext(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else f"cuda:{args.visible_gpus}"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = ExtSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter, step)

def train_ext(args, device_id):
    if (args.world_size > 1):
        train_multi_ext(args)
    else:
        train_single_ext(args, device_id)


def train_single_ext(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else f"cuda:{args.visible_gpus}"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    def train_iter_fct():
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = ExtSummarizer(args, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)

    logger.info(model)

    trainer = build_trainer(args, device_id, model, optim)
    trainer.train(train_iter_fct, args.train_steps)
