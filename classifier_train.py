"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    diffusion_defaults,
    classifier_defaults,
    create_classifier,
    create_gaussian_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict
from tqdm import tqdm
import torch.nn as nn


def main():
    args = create_argparser().parse_args()

    visible_gpus_list = [str(gpu_id) for gpu_id in args.gpus.split(",")]
    dist_util.setup_dist(visible_gpu_list=visible_gpus_list)
    logger.configure(dir=os.path.join(args.log_root, args.save_name))

    logger.log(args)
    logger.log("creating classifier and diffusion...")
    
    diffusion = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    classifier = create_classifier(
       **args_to_dict(args, classifier_defaults().keys()) 
    )

    if args.classifier_path:
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu"),
            strict=True,
        )

    classifier.to(dist_util.dev())
    device = next(classifier.parameters()).device

    if args.noised:
        schedule_sampler = create_named_schedule_sampler(
            args.schedule_sampler, diffusion
        )

    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        if dist.get_rank() == 0:
            logger.log(
                f"loading classifier from checkpoint: {args.resume_checkpoint}... at {resume_step} step"
            )
            classifier.load_state_dict(
                dist_util.load_state_dict(
                    args.resume_checkpoint, map_location=dist_util.dev()
                )
            )

    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(classifier.parameters())

    mp_trainer = MixedPrecisionTrainer(
        model=classifier, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    classifier = DDP(
        classifier,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.classifier_image_size,
        class_cond=True,
        random_crop=True,
        dataset_type=args.dataset_type,
        used_attributes=args.used_attributes,
        tot_class=args.tot_class
    )
    if args.val_data_dir:
        val_data = load_data(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            image_size=args.classifier_image_size,
            class_cond=True,
            dataset_type=args.dataset_type,
            used_attributes=args.used_attributes,
            tot_class=args.tot_class
        )
    else:
        val_data = None

    logger.log(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"].to(device)

        batch = batch.to(device)

        # Noisy images
        if args.noised:
            t, _ = schedule_sampler.sample(batch.shape[0], device)
            # print(t)
            batch = diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=device)

        for i, (sub_batch, sub_labels, sub_t) in enumerate(
                split_microbatches(args.microbatch, batch, labels, t)
        ):
            # print(sub_batch.shape)
            sig_op = nn.Sigmoid()
            bceloss = nn.BCELoss(size_average=False, reduce=False)

            logits = classifier(sub_batch, timesteps=sub_t)

            if args.dataset_type == 'celebahq':
                logits = sig_op(logits)

            if args.dataset_type == 'celebahq':
                loss = bceloss(logits, sub_labels).mean(dim=-1)

                a = th.ones_like(logits)
                b = a * 0.
                acc = th.where(logits > 0.5, a, b)
                acc = (acc == sub_labels).type(th.FloatTensor).mean(dim=-1)

            elif args.dataset_type == 'imagenet':
                loss = F.cross_entropy(logits, sub_labels, reduction="none")
            else:
                raise

            losses = {}
            losses[f"{prefix}_loss"] = loss.detach()

            if args.dataset_type == 'imagenet':
                losses[f"{prefix}_acc@1"] = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )
                losses[f"{prefix}_acc@5"] = compute_top_k(
                    logits, sub_labels, k=5, reduction="none"
                )

            elif args.dataset_type == 'celebahq':
                losses[f"{prefix}_acc@1"] = acc.detach()

            else:
                raise

            log_loss_dict(diffusion, sub_t, losses)
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in tqdm(range(args.iterations - resume_step)):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(data)
        mp_trainer.optimize(opt)

        if val_data is not None and not step % args.eval_interval:
            with th.no_grad():
                with classifier.no_sync():
                    classifier.eval()
                    forward_backward_log(val_data, prefix="val")
                    classifier.train()

        if not step % args.log_interval:
            logger.dumpkvs()
        if (
                step
                and dist.get_rank() == 0
                and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i: i + microbatch] if x is not None else None for x in args)


def save_files():
    dir_names = ['datasets', 'guided_diffusion', 'scripts', 'bash']
    file_names = ['classifier_sample.py', 'classifier_train.py', '*.bash']

    save_path = os.path.join(logger.get_dir(), 'saved_files')
    os.makedirs(save_path)

    for n in dir_names:
        os.system(f'cp -r {n} {save_path}')

    for n in file_names:
        os.system(f'cp {n} {save_path}')


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",

        log_interval=10000,
        eval_interval=100,
        save_interval=10000,

        log_root="",
        save_name="",
        gpus='3',

        tot_class=1000,
        dataset_type='imagenet',
        classifier_path="",

        used_attributes="",
        finetune_from_model_path = ""
    )
    defaults.update(classifier_defaults())
    defaults.update(diffusion_defaults())
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
