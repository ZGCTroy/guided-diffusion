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

        step_id = t[0].item() // 40
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            output = classifier(x_in, t)

            logits = output

            log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
            selected = log_probs[range(len(logits)), y.view(-1)]  # (B, )
            cond_grad = th.autograd.grad(selected.sum(), x_in)[0]

            # if t[0]>=840 and t[0]<1000:
            #     cond_grad = cond_grad * 10.0
            # else:
            #     cond_grad = cond_grad * args.classifier_scale
            cond_grad = cond_grad * args.classifier_scale

        with th.no_grad():

            probs = F.softmax(logits, dim=-1)  # (B, C)

            entropy = (-log_probs * probs).sum(dim=-1) / np.log(args.classifier_out_channels)  # (B, )
            entropy_scale = 1.0 / entropy  # (B, )
            original_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()

            batch_probability[step_id] = th.exp(selected)
            batch_probability_distribution[step_id] = probs
            batch_grad_norm[step_id] = original_grad_norm
            batch_entropy[step_id] = entropy
            batch_entropy_scale[step_id] = entropy_scale

            logger.log(
                '\n',
                't = ', t[0].detach(), '\n',
                '\t\t mean std median', '\n',
                '\t\t grad_norm =', original_grad_norm.mean(-1).detach(), original_grad_norm.std(-1).detach(), original_grad_norm.median(-1).values, '\n',
                '\t\t logits = ', selected.mean(-1).detach(), selected.std(-1).detach(), selected.median(-1).values, '\n',
            )

            if args.use_entropy_scale and (t[0] >= t_range_start and t[0] < t_range_end):
                updated_selected = args.classifier_scale * selected * entropy_scale  # (B, )
                cond_grad = cond_grad * entropy_scale.reshape(-1, 1, 1, 1).repeat(1, *cond_grad[0].shape)
                updated_grad_norm = th.norm(cond_grad, p=2, dim=(1, 2, 3), dtype=th.float32).detach()
                batch_updated_grad_norm[step_id] = updated_grad_norm

                logger.log(
                    '\t\t updated_selected = ', updated_selected.mean(-1).detach(), updated_selected.std(-1).detach(),
                    updated_selected.median(-1).values, '\n',
                    '\t\t entropy = ', entropy.mean(-1).detach(), entropy.std(-1).detach(), entropy.median(-1).values, '\n',
                    '\t\t entropy_scale = ', entropy_scale.mean(-1).detach(), entropy_scale.std(-1).detach(), entropy_scale.median(-1).values, '\n',
                    '\t\t updated_grad_norm = ',
                    updated_grad_norm.mean(-1).detach(), updated_grad_norm.std(-1).detach(), updated_grad_norm.median(-1).values, '\n',
                    '\n'
                )

                if latter_no_grad:
                    if t[0] >= 0 and t[0] < 840:
                        cond_grad = cond_grad * 0.0
                return cond_grad

            if latter_no_grad:
                if t[0] >= 0 and t[0] < 840:
                    cond_grad = cond_grad * 0.0

            return cond_grad

    def model_fn(x, t, y=None, t_range_start=0, t_range_end=1000, latter_no_grad=False):
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
        model_kwargs['latter_no_grad'] = False

        sample_fn = (
            diffusion.p_sample_loop_get_middle_img if not args.use_ddim else diffusion.ddim_sample_loop_get_middle_img
        )
        os.makedirs(
            os.path.join(logger.get_dir(), "class{}".format(args.selected_class)),
            exist_ok=True
        )
        for img_id in range(3):
            noise = th.randn((args.batch_size, 3, args.image_size, args.image_size), device=dist_util.dev())

            # 1
            args.classifier_scale = 10.0
            args.use_entropy_scale = False
            model_kwargs['latter_no_grad'] = True
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())

            sample, middle_imgs = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise=noise
            )  # middle_imgs : (steps, B, C, H, W)

            print(batch_probability)
            for i, img in enumerate(middle_imgs):
                # ax = plt.subplot(5,5,i+1)
                # ax.set_title('Time diffusion steps = {}'.format(960 - 40 * i))

                middle_imgs[i] = ((middle_imgs[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)

                middle_imgs[i] = 2 * (middle_imgs[i] / 255.) - 1.
                # middle_imgs[i] = middle_imgs[i].permute(0, 2, 3, 1)
                # middle_imgs[i] = middle_imgs[i].contiguous()
                print(middle_imgs[i].shape)

            print(th.cat(middle_imgs, dim=0).shape)
            utils.save_image(
                th.cat(middle_imgs, dim=0),
                os.path.join(logger.get_dir(), "class{}/middle_imgs_0-840_no_grad_{}.png".format(args.selected_class, img_id)),
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )

            # 2 baseline
            args.classifier_scale = 10.0
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
            args.use_entropy_scale = False
            model_kwargs['latter_no_grad'] = False
            sample, middle_imgs = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise=noise
            )  # middle_imgs : (steps, B, C, H, W)
            print(batch_probability)
            for i, img in enumerate(middle_imgs):
                # ax = plt.subplot(5,5,i+1)
                # ax.set_title('Time diffusion steps = {}'.format(960 - 40 * i))

                middle_imgs[i] = ((middle_imgs[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)

                middle_imgs[i] = 2 * (middle_imgs[i] / 255.) - 1.
                # middle_imgs[i] = middle_imgs[i].permute(0, 2, 3, 1)
                # middle_imgs[i] = middle_imgs[i].contiguous()
                print(middle_imgs[i].shape)

            print(th.cat(middle_imgs, dim=0).shape)
            utils.save_image(
                th.cat(middle_imgs, dim=0),
                os.path.join(logger.get_dir(), "class{}/middle_imgs_baseline_{}.png".format(args.selected_class, img_id)),
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )

            # 3
            args.classifier_scale = 6.0
            batch_probability = th.zeros((step_size, args.batch_size,), device=dist_util.dev())
            args.use_entropy_scale = True
            model_kwargs['latter_no_grad'] = False
            model_kwargs['t_range_start'] = 840
            model_kwargs['t_range_end'] = 1000

            sample, middle_imgs = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
                noise=noise
            )  # middle_imgs : (steps, B, C, H, W)
            print(batch_probability)

            for i, img in enumerate(middle_imgs):
                # ax = plt.subplot(5,5,i+1)
                # ax.set_title('Time diffusion steps = {}'.format(960 - 40 * i))

                middle_imgs[i] = ((middle_imgs[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)

                middle_imgs[i] = 2 * (middle_imgs[i] / 255.) - 1.
                # middle_imgs[i] = middle_imgs[i].permute(0, 2, 3, 1)
                # middle_imgs[i] = middle_imgs[i].contiguous()
                print(middle_imgs[i].shape)

            print(th.cat(middle_imgs, dim=0).shape)
            utils.save_image(
                th.cat(middle_imgs, dim=0),
                os.path.join(logger.get_dir(), "class{}/middle_imgs_entropyScale_{}.png".format(args.selected_class, img_id)),
                nrow=5,
                normalize=True,
                range=(-1, 1),
            )

        return

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
