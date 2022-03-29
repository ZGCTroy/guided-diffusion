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
from guided_diffusion.sample_util import save_samples
from tqdm import tqdm


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
    print(model)
    diffusion = create_gaussian_diffusion(
        **args_to_dict(args, diffusion_defaults().keys())
    )
    if args.model_path:
        logger.log("loading model from {}".format(args.model_path))
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

    def cond_fn(x, t, y=None, prior_variance=1.0, t_range_start=0, t_range_end=1000):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = classifier(x_in, t)

            logits = output

            log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
            selected = log_probs[range(len(logits)), y.view(-1)]  # (B, )
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]

            cond_grad = cond_grad * args.classifier_scale

            if args.use_cond_range_scale:
                if t[0] >= 0 and t[0] < 500:
                    cond_grad = cond_grad * 3

        with th.no_grad():

            if args.use_probability_scale and (t[0] >= t_range_start and t[0] < t_range_end):
                probability = F.softmax(logits, dim=-1)[range(len(logits)), y.view(-1)]
                probability_scale = 1.0 / (1.0 - probability)

                original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                cond_grad = cond_grad * probability_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                updated_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()

                logger.log(
                    't = ', t[0].detach(), '\n',
                    '\t\t mean std median', '\n',
                    '\t\t probability = ', probability.mean(-1).detach(), probability.std(-1).detach(), probability.median(-1).values, '\n',
                    '\t\t probability_scale = ', probability_scale.mean(-1).detach(), probability_scale.std(-1).detach(), probability_scale.median(-1).values, '\n',
                    '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                    '\t\t updated_grad_norm = ', updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )
                return cond_grad

            if args.use_entropy_scale and (t[0] >= t_range_start and t[0] < t_range_end):

                probs = F.softmax(logits, dim=-1)  # (B, C)
                entropy = (-log_probs * probs).sum(dim=-1)  # (B, )
                if args.use_normalized_entropy_scale:
                    entropy_scale = 1.0 / (entropy / np.log(args.classifier_out_channels))  # (B, )
                else:
                    entropy_scale = 1.0 / entropy

                updated_selected = args.classifier_scale * selected * entropy_scale  # (B, )
                original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                cond_grad = cond_grad * entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                updated_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()

                logger.log(
                    't = ', t[0].detach(), '\n',
                    '\t\t mean std median', '\n',
                    '\t\t original_selected = ', selected.mean(-1).detach(), selected.std(-1).detach(), selected.median(-1).values, '\n',
                    '\t\t entropy = ', entropy.mean(-1).detach(), entropy.std(-1).detach(), entropy.median(-1).values, '\n',
                    '\t\t entropy_scale = ', entropy_scale.mean(-1).detach(), entropy_scale.std(-1).detach(), entropy_scale.median(-1).values, '\n',
                    '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                    '\t\t updated_grad_norm = ',
                    updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )
                return cond_grad

            original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
            logger.log(
                't = ', t[0].detach(), '\n',
                '\t\t mean std median', '\n',
                '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n'
                                                                                                                                                     '\t\t logits = ', selected.mean(-1).detach(),
                selected.std(-1).detach(), selected.median(-1).values, '\n',
                '\n'
            )

            return cond_grad

    def model_fn(x, t, y=None, t_range_start=0, t_range_end=1000):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
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

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} / {args.num_samples} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])

        if args.selected_class == -1:
            out_path = os.path.join(logger.get_dir(), "scale{}_steps{}_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, shape_str))
        else:
            out_path = os.path.join(logger.get_dir(), "scale{}_steps{}_class{}_samples_{}.npz".format(args.classifier_scale, args.timestep_respacing, args.selected_class, shape_str))

        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

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
        use_probability_scale=False,
        use_normalized_entropy_scale=True
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
