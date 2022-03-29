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

from torchvision import utils
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

    def cond_fn(x, t, y=None, prior_variance=1.0, t_range_start=0, t_range_end=1000, latter_no_grad=False):

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
                batch_updated_grad_norm[step_id] = updated_grad_norm

                logger.log(

                    '\t\t updated_grad_norm = ',
                    updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )

            if latter_no_grad:
                if t[0] >= 0 and t[0] < 640:
                    cond_grad = cond_grad * 0.0

            return cond_grad

    def model_fn(x, t, y=None, t_range_start=0, t_range_end=1000, latter_no_grad=False):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")

    id = 0
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
    model_kwargs['latter_no_grad'] = False

    sample_fn = (
        diffusion.p_sample_loop_get_middle_img if not args.use_ddim else diffusion.ddim_sample_loop_get_middle_img
    )
    os.makedirs(
        os.path.join(logger.get_dir(), "class{}".format(args.selected_class)),
        exist_ok=True
    )
    noise = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev())

    # 1 no_grad
    for method in ['baseline', 'ECT+EDS']:
        if method == '0-640_no_grad':
            args.classifier_scale = 10.0
            args.use_entropy_scale = False
            model_kwargs['latter_no_grad'] = True
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
        elif method == 'baseline':
            args.classifier_scale = 10.0
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
            args.use_entropy_scale = False
            model_kwargs['latter_no_grad'] = False
        elif method == 'EDS':
            args.classifier_scale = 6.0
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
            args.use_entropy_scale = True
            model_kwargs['latter_no_grad'] = False
            model_kwargs['t_range_start'] = 0
            model_kwargs['t_range_end'] = 1000

        elif method == 'ECT+EDS':
            if args.timestep_respacing == 'ddim25':
                args.classifier_scale = 6.0
            else:
                args.classifier_scale = 4.0
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
            args.use_entropy_scale = True
            model_kwargs['latter_no_grad'] = False
            model_kwargs['t_range_start'] = 0
            model_kwargs['t_range_end'] = 1000

            classifier = create_classifier(
                **args_to_dict(args, classifier_defaults().keys())
            )

            if args.classifier_path:
                ECT_EDS_classifier_path = '/workspace/mnt/storage/guangcongzheng/zju_zgc/guided-diffusion/log/imagenet1000_classifier256x256_channel128_entropyConstraintTrain0.2/model500000.pt'
                logger.log("loading classifier from {}".format(ECT_EDS_classifier_path))
                classifier.load_state_dict(
                    dist_util.load_state_dict(ECT_EDS_classifier_path, map_location="cpu"),
                    strict=True
                )
            classifier.to(dist_util.dev())
            if args.classifier_use_fp16:
                classifier.convert_to_fp16()
            classifier.eval()

        _, sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            noise=noise
        )  # middle_imgs : (steps, B, C, H, W), sample: (B,C,H,W)

        sample = th.stack(sample, dim=0)
        print(sample.shape)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = 2 * (sample / 255.) - 1.
        save_path = os.path.join(logger.get_dir(), "class{}/imgs_{}.png".format(args.selected_class, method))
        print('save to {}'.format(save_path))
        utils.save_image(
            sample[-1],
            save_path,
            nrow=4,
            normalize=True,
            range=(-1, 1),
        )
        metainfo_out_path = os.path.join(logger.get_dir(), "class{}/metainfo_{}.npz".format(args.selected_class, method))
        np.savez(
            metainfo_out_path,
            batch_image=np.array(sample.cpu()),
            batch_label=np.array(classes.cpu()),
            batch_grad_norm=np.array(batch_grad_norm.cpu()),
            batch_updated_grad_norm=np.array(batch_updated_grad_norm.cpu()),
            batch_probability=np.array(batch_probability.cpu()),
            batch_entropy=np.array(batch_entropy.cpu()),
            batch_entropy_scale=np.array(batch_entropy_scale.cpu()),
            batch_probability_distribution=np.array(batch_probability_distribution.cpu()),
        )


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
