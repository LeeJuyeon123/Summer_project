# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
import time
import utils
from monitoring import log_mem

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,max_iterations=None):

    model.train(set_training_mode)
    original_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    key_convert_time = 0.0
    model_forward_pass_time = 0.0
    backward_time = 0.0
    update_time = 0.0  
    current_iterations = 0
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        
        if current_iterations >= max_iterations:
            print("Computation budget exhausted. Logging metrics...")
            # Log using MetricLogger
            print("Averaged stats up to iteration {}: ".format(current_iterations), metric_logger)
            break
        
        input = input.to(device)
        target = target.to(device)

        if isinstance(device, torch.device) and device.type == "cuda":
            dev_id = device.index if device.index is not None else torch.cuda.current_device()
        elif isinstance(device, int):  # device가 int로 들어오는 코드베이스라면
            dev_id = device
        else:
            dev_id = None  # CPU라면 None

        torch.cuda.reset_peak_memory_stats(dev_id)

        key_start_time = torch.cuda.Event(enable_timing=True)
        key_start_time.record()
        with torch.no_grad():
            output = original_model(input)
            cls_features = output['pre_logits']
            key_end_time = torch.cuda.Event(enable_timing=True)
            key_end_time.record()
        
        torch.cuda.synchronize()
        key_convert_time += key_start_time.elapsed_time(key_end_time)

        # Model Forward Pass
        model_start_time = torch.cuda.Event(enable_timing=True)
        model_start_time.record()

        #forward 직전
        log_mem("before_forward", dev_id)
        model.swap_pack_ms = 0.0
        model.swap_unpack_ms = 0.0

        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']

        #forward 직후
        log_mem("after_forward_pack", dev_id)
        print(f"[SWAP-TIME] Step pack total: {model.swap_pack_ms:.3f} ms")
        if hasattr(model, "swap_dbg"):
            model.swap_dbg.snapshot()

        model_end_time = torch.cuda.Event(enable_timing=True)
        model_end_time.record()

        torch.cuda.synchronize()
        model_forward_pass_time += model_start_time.elapsed_time(model_end_time)
        
        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        backward_start_time = torch.cuda.Event(enable_timing=True)
        backward_start_time.record()
        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)

        #backward 직전
        log_mem("before_backward", dev_id)

        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() #여기서 unpack 수행 (?)

        #backward(unpack) 직후
        log_mem("after_backward", dev_id)
        print(f"[SWAP-TIME] Step unpack total: {model.swap_unpack_ms:.3f} ms")
        if hasattr(model, "swap_dbg"):
            model.swap_dbg.snapshot()

        backward_end_time = torch.cuda.Event(enable_timing=True)
        backward_end_time.record()
        torch.cuda.synchronize()
        backward_time += backward_start_time.elapsed_time(backward_end_time)

        # Model Update
        model_update_start_time = torch.cuda.Event(enable_timing=True)
        model_update_start_time.record()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        #step이후
        log_mem("after_step", dev_id)

        model_update_end_time = torch.cuda.Event(enable_timing=True)
        model_update_end_time.record()
        torch.cuda.synchronize()
        update_time += model_update_start_time.elapsed_time(model_update_end_time)

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
       
        current_iterations += 1
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # task_total_iterations += current_iterations
    num_iterations = len(data_loader)
    avg_key_convert_time = key_convert_time / num_iterations
    avg_model_forward_time = model_forward_pass_time / num_iterations 
    avg_backward_time = backward_time / num_iterations
    avg_update_time = update_time / num_iterations
    
    print("Averaged stats for epoch {}: ".format(epoch+1), metric_logger)
    print(f"Avg Prompt Selection Time for epoch {epoch+1}: {avg_key_convert_time:.4f} ms")
    print(f"Avg Model Forward Time for epoch {epoch+1}: {avg_model_forward_time:.4f} ms")
    print(f"Avg Backward Pass Time for epoch {epoch+1}: {avg_backward_time:.4f} ms")
    print(f"Avg Model Update Time for epoch {epoch+1}: {avg_update_time:.4f} ms")   


    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['avg_key_convert_time'] = avg_key_convert_time
    stats['avg_model_forward_time'] = avg_model_forward_time
    stats['avg_backward_time'] = avg_backward_time
    stats['avg_update_time'] = avg_update_time

    return stats



@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                mask = torch.tensor(mask, dtype=torch.int64).to(device)
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
                    class_mask=None, args=None):
   
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    original_budget = args.budget_per_task

    for task_id in range(args.num_tasks):
        
        total_iterations_per_epoch = len(data_loader[task_id]['train'])
        args.budget_per_task = original_budget  # reset the budget for each task
        print(f"Computation Budget for Task {task_id+1}: {original_budget}")
        
        # Initialize accumulators for the time metrics
        total_key_convert_time = 0.0
        total_model_forward_time = 0.0
        total_backward_time = 0.0
        total_update_time = 0.0

        # Transfer previous learned prompt params to the new prompt
        if args.use_prompt:
            if args.prompt_pool and args.shared_prompt_pool:
                if task_id > 0:
                    prev_start = (task_id - 1) * args.top_k
                    prev_end = task_id * args.top_k
                    cur_start = prev_end
                    cur_end = (task_id + 1) * args.top_k
                    if (prev_end > args.size) or (cur_end > args.size):
                        pass
                    else:
                        cur_idx = (slice(cur_start, cur_end))
                        prev_idx = (slice(prev_start, prev_end))
                        with torch.no_grad():
                            if args.distributed:
                                model.module.prompt.prompt.grad.zero_()
                                model.module.prompt.prompt[cur_idx] = model.module.prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.module.parameters()
                            else:
                                model.prompt.prompt.grad.zero_()
                                model.prompt.prompt[cur_idx] = model.prompt.prompt[prev_idx]
                                optimizer.param_groups[0]['params'] = model.parameters()

            # Transfer previous learned prompt param keys to the new prompt
            if args.prompt_pool and args.shared_prompt_key:
                if task_id > 0:                    
                    prev_start = (task_id - 1) * args.top_k
                    prev_end = task_id * args.top_k
                    cur_start = prev_end
                    cur_end = (task_id + 1) * args.top_k
                    with torch.no_grad():
                        if args.distributed:
                            model.module.prompt.prompt_key.grad.zero_()
                            model.module.prompt.prompt_key[cur_idx] = model.module.prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.prompt.prompt_key.grad.zero_()
                            model.prompt.prompt_key[cur_idx] = model.prompt.prompt_key[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            max_iterations = min(args.budget_per_task, total_iterations_per_epoch)

            if max_iterations <= 0:
                break
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                          device=device, epoch=epoch, max_norm=args.clip_grad, 
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, max_iterations=max_iterations)
            
            args.budget_per_task -= max_iterations
            
            # Accumulate timings from the current epoch
            total_key_convert_time += train_stats['avg_key_convert_time'] 
            total_model_forward_time += train_stats['avg_model_forward_time'] 
            total_backward_time += train_stats['avg_backward_time'] 
            total_update_time += train_stats['avg_update_time'] 

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        
        total_avg_key_convert_time = total_key_convert_time / (args.epochs)
        total_avg_model_forward_time = total_model_forward_time / (args.epochs)
        
        total_avg_forward_time = (total_avg_key_convert_time + total_avg_model_forward_time)
        total_avg_backward_time = total_backward_time / (args.epochs)
        
        total_avg_update_time = total_update_time / (args.epochs)
        total_avg_iteration_time = total_avg_forward_time + total_avg_backward_time + total_avg_update_time

        print("\n----- Total Training Stats -----")
        print(f"Total Avg Prompt Selection Time for Task {task_id+1}: {total_avg_key_convert_time:.4f} ms")
        print(f"Total Avg Model Forward Pass Time for Task {task_id+1}: {total_avg_model_forward_time:.4f} ms")
        print(f"Total Avg Forward Pass Time for Task {task_id+1}: {total_avg_forward_time:.4f} ms")
        print(f"Total Avg Backward Pass Time for Task {task_id+1}: {total_avg_backward_time:.4f} ms")
        print(f"Total Avg Model Update Time for Task {task_id+1}: {total_avg_update_time:.4f} ms")
        print(f"Total Avg Iteration Time for Task {task_id+1}: {total_avg_iteration_time:.4f} ms")