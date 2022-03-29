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
import matplotlib.pyplot as plt
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

    step_size = 25 if args.timestep_respacing == 'ddim25' else int(args.timestep_respacing)
    batch_grad_norm = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_updated_grad_norm = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_entropy = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_entropy_scale = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
    batch_probability_distribution = th.zeros((step_size, args.batch_size, args.classifier_out_channels,), device=dist_util.dev())

    def cond_fn(x, t, y=None, prior_variance=1.0, t_range_start=0, t_range_end=1000):

        assert y is not None

        if step_size == 25:
            step_id = t[0].item() // 40
        elif step_size == 250:
            step_id = t[0].item() // 4

        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = classifier(x_in, t)

            logits = output

            log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
            selected = log_probs[range(len(logits)), y.view(-1)]  # (B, )
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]

            cond_grad = cond_grad * args.classifier_scale

        with th.no_grad():

            probs = F.softmax(logits, dim=-1)  # (B, C)
            entropy = (-log_probs * probs).sum(dim=-1) / np.log(args.classifier_out_channels)  # (B, )
            entropy_scale = 1.0 / entropy  # (B, )
            original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
            selected_probability = probs[range(len(logits)), y.view(-1)]

            batch_probability[step_id] = selected_probability
            batch_probability_distribution[step_id] = probs
            batch_grad_norm[step_id] = original_grad_norm
            batch_entropy[step_id] = entropy
            batch_entropy_scale[step_id] = entropy_scale

            logger.log(
                '\n',
                't = ', t[0].detach(), '\n',
                '\t\t mean std median', '\n',
                '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                '\t\t probability = ', selected_probability.mean(-1).detach(), selected_probability.std(-1).detach(), selected_probability.median(-1).values, '\n',
                '\t\t entropy = ', entropy.mean(-1).detach(), entropy.std(-1).detach(), entropy.median(-1).values, '\n',
                '\t\t entropy_scale = ', entropy_scale.mean(-1).detach(), entropy_scale.std(-1).detach(), entropy_scale.median(-1).values, '\n',
            )

            if args.use_entropy_scale and (t[0] >= t_range_start and t[0] < t_range_end):
                cond_grad = cond_grad * entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                updated_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                batch_updated_grad_norm[step_id]=updated_grad_norm

                logger.log(

                    '\t\t updated_grad_norm = ',
                    updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )

                return cond_grad

            return cond_grad

    def model_fn(x, t, y=None, t_range_start=0, t_range_end=1000):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

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
        model_kwargs = {}
        if args.selected_class == -1:
            classes = th.randint(
                low=0, high=args.classifier_out_channels, size=(args.batch_size,), device=dist_util.dev()
            )
        else:
            classes = th.randint(
                low=args.selected_class, high=args.selected_class + 1, size=(args.batch_size,), device=dist_util.dev()
            )

        model_kwargs["y"] = classes
        model_kwargs['t_range_start'] = args.t_range_start
        model_kwargs['t_range_end'] = args.t_range_end

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = get_gathered_item(sample)
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])

        gathered_labels = get_gathered_item(classes)
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

        if args.selected_class == -1:
            metainfo_out_path = os.path.join(logger.get_dir(), "metainfo_scale{}_steps{}_class0-999_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, shape_str))
        else:
            metainfo_out_path = os.path.join(logger.get_dir(), "metainfo_scale{}_steps{}_class{}_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, args.selected_class, shape_str))

        np.savez(metainfo_out_path,all_grad_norm=all_grad_norm, all_updated_grad_norm=all_updated_grad_norm,
                 all_probability=all_probability, all_entropy=all_entropy, all_entropy_scale=all_entropy_scale,
                 all_probability_distribution=all_probability_distribution)

        logger.log(f"saving to {metainfo_out_path}")


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
