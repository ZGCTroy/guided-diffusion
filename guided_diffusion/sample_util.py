import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from . import dist_util, logger
import numpy as np
import os
from torchvision import utils

def quick_sample(classifier, model, diffusion,
                 use_ddim=False,
                 num_samples=64,
                 batch_size=32,
                 image_size=256,
                 clip_denoised=True,
                 classifier_scale=10.,
                 classifier_out_channels=10,
                 class_cond=False,
                 current_steps=0,
                 ):
    '''
    added 12.2
    '''
    logger.log("sampling...")

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if class_cond else None)

    save_path = os.path.join(logger.get_dir(), 'predicted_imgs')
    all_images = []
    all_labels = []
    idx = 0
    while len(all_images) * batch_size < num_samples:
        idx += 1
        model_kwargs = {}
        classes = th.randint(
            low=0, high=classifier_out_channels, size=(batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (batch_size, 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )

        if dist.get_rank() == 0:
            utils.save_image(
                sample,
                os.path.join(save_path, f"{str(current_steps).zfill(5)}_{idx}.png"),
                nrow=batch_size,
                normalize=True,
                range=(-1, 1),
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
        logger.log(f"created {len(all_images) * batch_size} / {num_samples} samples")

    save_arr = np.concatenate(all_images, axis=0)
    save_label = np.concatenate(all_labels, axis=0)

    eva_npz_path = os.path.join(save_path, f'images{str(current_steps).zfill(5)}.npz')
    if dist.get_rank() == 0:
        np.savez(eva_npz_path, save_arr, save_label)

    logger.log("sampling complete")
    return eva_npz_path


def save_files():
    dir_names = ['datasets', 'guided_diffusion', 'scripts', 'bash']
    file_names = ['classifier_*.py']
    save_path = os.path.join(logger.get_dir(), 'predicted_imgs')
    os.makedirs(save_path, exist_ok=True)

    save_path = os.path.join(logger.get_dir(), 'saved_files')
    os.makedirs(save_path, exist_ok=True)

    for n in dir_names:
        os.system(f'cp -r {n} {save_path}')

    for n in file_names:
        os.system(f'cp {n} {save_path}')



def save_samples(img_arr, label_arr, num_classes,
                per_class_ub=128,
                save_batch_size=32,    
                nrow=8, log_dir=None, sample_dir=""):
    # num_classes = label_arr.max()
    # sorted_classes = {x: i for i, x in enumerate(sorted(set(label_arr.tolist())))}
    if not log_dir:
        log_dir = logger.get_dir()

    for i in range(num_classes):
        # class_imgs = []
        idx = np.where(label_arr==i)[0]

        print(idx, 'idx',i, len(idx))
        if len(idx) == 0:
            # print('yyyyyy')
            continue
        # else:
        # print(img_arr[idx].shape)
        # continue
        
        idx = idx[:per_class_ub]

        img = img_arr[idx].transpose(0, 3, 1, 2)
        img = 2 * (th.from_numpy(img) / 255.) - 1.
        # class_imgs.append(img)

        save_path = os.path.join(log_dir, sample_dir, f'class{i}')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=False)
        
        # saved_imgs = th.cat(class_imgs, dim=0)
        batches = th.split(img, save_batch_size, dim=0)
        for idx,batch in enumerate(batches):
            utils.save_image(
                batch,
                os.path.join(save_path, f"imgs_{idx}.png"),
                nrow=nrow,
                normalize=True,
                range=(-1, 1),
            )

    return 
