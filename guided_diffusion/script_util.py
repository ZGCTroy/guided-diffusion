import argparse
import inspect

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel, UNetModel, EncoderUNetModel

def diffusion_defaults():
    """
    Defaults for image and classifier training.
    """
    return dict(
        diffusion_steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
    )

def model_defaults():
    res = dict(
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        channel_mult="",
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        num_classes=1000
    )
    return res

def classifier_defaults():
    """
    Defaults for classifier models.
    """
    res = dict(
        classifier_image_size=64,
        classifier_num_channels=128,
        classifier_num_res_blocks=2,
        classifier_channel_mult="",
        classifier_use_checkpoint=False,
        classifier_attention_resolutions="32,16,8",
        classifier_num_heads=4,
        classifier_num_head_channels=64,
        classifier_num_heads_upsample=-1,
        classifier_use_scale_shift_norm=True,
        classifier_dropout=0.0,
        classifier_resblock_updown=False,
        classifier_use_fp16=False,
        classifier_use_new_attention_order=False,
        classifier_pool="attention",
        classifier_out_channels = 1000
    )

    return res

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    num_classes
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(num_classes if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_classifier(
    classifier_image_size,
    classifier_num_channels,
    classifier_num_res_blocks,
    classifier_channel_mult,
    classifier_use_fp16,
    classifier_num_heads,
    classifier_num_heads_upsample,
    classifier_num_head_channels,
    classifier_attention_resolutions,
    classifier_dropout,
    classifier_use_checkpoint,
    classifier_use_scale_shift_norm,
    classifier_resblock_updown,
    classifier_use_new_attention_order,
    classifier_pool,
    classifier_out_channels,

):
    if classifier_channel_mult == "":
        if classifier_image_size == 512:
            classifier_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif classifier_image_size == 256:
            classifier_channel_mult = (1, 1, 2, 2, 4, 4)
        elif classifier_image_size == 128:
            classifier_channel_mult = (1, 1, 2, 3, 4)
        elif classifier_image_size == 64:
            classifier_channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {classifier_image_size}")
    else:
        classifier_channel_mult = tuple(int(ch_mult) for ch_mult in classifier_channel_mult.split(","))

    attention_ds = []
    for res in classifier_attention_resolutions.split(","):
        attention_ds.append(classifier_image_size // int(res))

    return EncoderUNetModel(
        image_size=classifier_image_size,
        in_channels=3,
        model_channels=classifier_num_channels,
        out_channels=classifier_out_channels,
        num_res_blocks=classifier_num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=classifier_dropout,
        channel_mult=classifier_channel_mult,
        use_checkpoint=classifier_use_checkpoint,
        use_fp16=classifier_use_fp16,
        num_heads=classifier_num_heads,
        num_head_channels=classifier_num_head_channels,
        num_heads_upsample=classifier_num_heads_upsample,
        use_scale_shift_norm=classifier_use_scale_shift_norm,
        resblock_updown=classifier_resblock_updown,
        pool=classifier_pool,
        use_new_attention_order=classifier_use_new_attention_order
    )

def create_gaussian_diffusion(
    *,
    diffusion_steps,
    learn_sigma,
    sigma_small,
    noise_schedule,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    timestep_respacing,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
