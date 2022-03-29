"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from guided_diffusion.image_datasets import load_data
import matplotlib.pyplot as plt

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_defaults,
    diffusion_defaults,
    classifier_defaults,
    create_model,
    create_gaussian_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.sample_util import save_samples
from tqdm import tqdm


def get_gathered_item(x):
    gathered_x = [th.zeros_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_x, x)
    return gathered_x


def main():
    args = create_argparser().parse_args()

    visible_gpus_list = []
    if args.gpus:
        visible_gpus_list = [str(gpu_id) for gpu_id in args.gpus.split(",")]
    dist_util.setup_dist(visible_gpu_list=visible_gpus_list, local_rank=args.local_rank)

    logger.configure(dir=os.path.join(args.log_root, args.save_name))
    logger.log(args)

    logger.log("creating model and diffusion...")
    model = create_model(
        **args_to_dict(args, model_defaults().keys())
    )
    diffusion = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    if args.model_path:
        model.load_state_dict(
            dist_util.load_state_dict(args.model_path, map_location="cpu"),
            strict=True
        )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    classifier = create_classifier(
        **args_to_dict(args, classifier_defaults().keys())
    )

    if args.classifier_path:
        logger.log("loading classifier from {}".format(args.classifier_path))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu"),
            strict=True
        )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    step_size = 25 if args.timestep_respacing == 'ddim25' else args.timestep_respacing
    batch_grad_norm = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_updated_grad_norm = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_entropy = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_entropy_scale = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_probability_distribution = th.zeros((step_size, args.batch_size, args.classifier_out_channels,), device=dist_util.dev())

    def cond_fn(x, t, y=None, prior_variance=1.0, t_range_start=0, t_range_end=1000):
        assert y is not None

        step_id = t[0].item() // 40
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = classifier(x_in, t)
            logits = output

            log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
            selected = log_probs[range(len(logits)), y.view(-1)]  # (B, )
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]
            cond_grad = cond_grad * args.classifier_scale

        with th.no_grad():
            if args.use_cond_range_scale:
                if t[0] >= 600 and t[0] < 800:
                    cond_grad = cond_grad * 3

            probs = F.softmax(logits, dim=-1)  # (B, C)
            entropy = (-log_probs * probs).sum(dim=-1)  # (B, )
            entropy_scale = 1.0 / (entropy / np.log(args.classifier_out_channels))  # (B, )
            original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()

            batch_probability[step_id] = probs[range(len(logits)), y.view(-1)]
            batch_probability_distribution[step_id] = probs
            batch_grad_norm[step_id] = original_grad_norm
            batch_entropy[step_id] = entropy
            batch_entropy_scale[step_id] = entropy_scale

            logger.log(
                '\n',
                't = ', t[0].detach(), '\n',
                '\t\t mean std median', '\n',
                '\t\t grad_norm =', original_grad_norm.mean(-1).detach(),
                original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                '\t\t logits = ', selected.mean(-1).detach(),
                selected.std(-1).detach(), selected.median(-1).values, '\n',
                '\t\t entropy = ', entropy.mean(-1).detach(), entropy.std(-1).detach(), entropy.median(-1).values, '\n',
                '\t\t entropy_scale = ', entropy_scale.mean(-1).detach(), entropy_scale.std(-1).detach(), entropy_scale.median(-1).values, '\n',
            )

            if args.use_entropy_scale and (t[0] >= t_range_start and t[0] < t_range_end):
                cond_grad = cond_grad * entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                updated_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                batch_updated_grad_norm[step_id].append(updated_grad_norm)

                logger.log(
                    '\t\t updated_grad_norm = ', updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )

            return cond_grad

    def model_fn(x, t, y=None, t_range_start=0, t_range_end=1000):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.classifier_image_size,
        class_cond=True,
        random_crop=True,
        dataset_type=args.dataset_type,
        used_attributes=args.used_attributes,
        tot_class=args.tot_class,
        imagenet200_class_list_file_path=args.imagenet200_class_list_file_path,
        celeba_attribures_path=args.celeba_attribures_path
    )

    logger.log("sampling...")
    all_images, all_labels = [], []
    all_grad_norm = []
    all_updated_grad_norm = []
    all_probability = []
    all_entropy = []
    all_entropy_scale = []
    all_probability_distribution = []

    id = 0


    while len(all_images) * args.batch_size < args.num_samples:
        id += 1
        batch, extra = next(data_loader)
        labels = extra["y"].to(dist_util.dev())  # labels: (B, )
        batch = batch.to(dist_util.dev())  # batch:  (B, C, H, W)

        for t in range(24, -1, -1):
            batch_t = th.ones(batch.shape[0], dtype=th.long, device=dist_util.dev()) * t
            diffused_batch = diffusion.q_sample(batch, batch_t)

            cond_grad = cond_fn(
                x=diffused_batch,
                t=batch_t * 40,
                y=labels,
                prior_variance=1.0,
                t_range_start=0,
                t_range_end=1000
            )

        gathered_samples = get_gathered_item(diffused_batch)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = get_gathered_item(labels)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])

        logger.log(f"created {len(all_images) * args.batch_size} / {args.num_samples} samples")

        gathered_batch_grad_norm = get_gathered_item(batch_grad_norm)
        gathered_batch_updated_grad_norm = get_gathered_item(batch_updated_grad_norm)
        gathered_batch_probability = get_gathered_item(batch_probability)
        gathered_batch_entropy = get_gathered_item(batch_entropy)
        gathered_batch_entropy_scale = get_gathered_item(batch_entropy_scale)
        gathered_batch_probability_distribution = get_gathered_item(batch_probability_distribution)

        all_grad_norm.extend([x.cpu().numpy() for x in gathered_batch_grad_norm])
        all_updated_grad_norm.extend([x.cpu().numpy() for x in gathered_batch_updated_grad_norm])
        all_probability.extend([x.cpu().numpy() for x in gathered_batch_probability])
        all_entropy.extend([x.cpu().numpy() for x in gathered_batch_entropy])
        all_entropy_scale.extend([x.cpu().numpy() for x in gathered_batch_entropy_scale])
        all_probability_distribution.extend([x.cpu().numpy() for x in gathered_batch_probability_distribution])

    if dist.get_rank() == 0:
        arr = np.concatenate(all_images, axis=0)
        arr = arr[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]

        all_grad_norm = np.concatenate(all_grad_norm, axis=1)[:args.num_samples]
        all_updated_grad_norm = np.concatenate(all_updated_grad_norm, axis=1)[:args.num_samples]
        all_probability = np.concatenate(all_probability, axis=1)[:args.num_samples]
        all_entropy = np.concatenate(all_entropy, axis=1)[:args.num_samples]
        all_entropy_scale = np.concatenate(all_entropy_scale, axis=1)[:args.num_samples]
        all_probability_distribution = np.concatenate(all_probability_distribution, axis=1)[:args.num_samples]

        shape_str = "x".join([str(x) for x in arr.shape])

        out_path = os.path.join(logger.get_dir(), "scale{}_steps{}_class0-{}_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, args.tot_class - 1, shape_str))

        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

        metainfo_out_path = os.path.join(logger.get_dir(), "metainfo_scale{}_steps{}_class0-{}_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, args.tot_class - 1, shape_str))

        np.savez(metainfo_out_path, all_grad_norm=all_grad_norm, all_updated_grad_norm=all_updated_grad_norm,
                 all_probability=all_probability, all_entropy=all_entropy, all_entropy_scale=all_entropy_scale,
                 all_probability_distribution=all_probability_distribution)

        # sample_dir = "images_" + "scale{}_steps{}_sample{}".format(args.classifier_scale, args.timestep_respacing, args.num_samples)
        # save_samples(arr, label_arr, args.classifier_out_channels, sample_dir=sample_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        log_root="",
        save_name="",
        gpus="",
        t_range_start=0,
        t_range_end=1000,

        use_entropy_scale=False,

        expected_classifier_gradient_value=-1.0,

        selected_class=-1,
        use_cond_range_scale=False,

        data_dir="",
        val_data_dir="",
        tot_class=1000,
        dataset_type='imagenet-1000',
        used_attributes="",
        imagenet200_class_list_file_path="",
        celeba_attribures_path="",
    )
    defaults.update(diffusion_defaults())
    defaults.update(model_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
