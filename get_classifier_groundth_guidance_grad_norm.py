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
    step_size = int(step_size)


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


    id = 0
    with th.no_grad():
        for idx in range(0, args.num_samples, args.batch_size):
            id += 1
            batch, extra = next(data_loader)
            labels = extra["y"].to(dist_util.dev())  # labels: (B, )
            batch = batch[0].view(1,3,256,256)
            batch = th.repeat_interleave(batch, dim=0, repeats=args.batch_size)
            batch = batch.to(dist_util.dev())  # batch:  (B, C, H, W)


            model_kwargs = {}
            model_kwargs["y"] = labels
            model_kwargs['t_range_start'] = args.t_range_start
            model_kwargs['t_range_end'] = args.t_range_end

            for t in range(249, -1, -10):
                batch_t = th.ones(batch.shape[0], dtype=th.long, device=dist_util.dev()) * t
                diffused_batch = diffusion.q_sample(batch, batch_t)
                next_diffused_batch = diffusion.q_sample(batch, batch_t-1)

                next_x = diffusion.p_mean_variance(
                    model_fn,
                    diffused_batch,
                    batch_t,
                    clip_denoised=args.clip_denoised,
                    denoised_fn=None,
                    model_kwargs=model_kwargs,
                )

                if dist.get_rank() == 0:
                    next_predicted_x_mean = next_x['mean']          # (B, C, W, H)
                    next_predicted_x_variance = next_x["variance"]  # (B, C, W, H)

                    next_diffused_batch = next_diffused_batch.mean(0)       # ( C, W, H)
                    next_diffused_batch = th.repeat_interleave(next_diffused_batch.view(1,3,256,256), dim=0, repeats=args.batch_size) # (B, C, W, H)

                    diff = next_diffused_batch - next_predicted_x_mean      # (B, C, W, H)
                    ground_truth_cond_grad = diff / next_predicted_x_variance    # (B, C, W, H)
                    print(diff.mean())
                    print(ground_truth_cond_grad.mean())
                    norm = th.norm(ground_truth_cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).mean()

                    print(t, norm)

                # if dist.get_rank() == 0:
                #     next_predicted_x_mean = next_x['mean'].mean(0)           # (C, W, H)
                #     next_diffused_batch = next_diffused_batch.mean(0)        # (C, W, H)
                #     next_predicted_x_variance = next_x["variance"].mean(0)  # (C, W, H)
                #     diff = next_diffused_batch - next_predicted_x_mean      # (C, W, H)
                #     ground_truth_cond_grad = diff / next_predicted_x_variance    # (C, W, H)
                #     print(ground_truth_cond_grad.shape)
                #     norm = th.norm(ground_truth_cond_grad, p=2, dim=(0, 1, 2), dtype=th.float32)
                #
                #     print(t, next_diffused_batch.mean().item(), next_predicted_x_mean.mean().item(), next_predicted_x_variance.mean().item())
                #     print('diff = ', diff.mean())
                #     print('ground truth cond grad', ground_truth_cond_grad.mean())
                #     print('norm = ', norm)

            print()

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
