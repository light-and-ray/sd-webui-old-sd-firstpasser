import math
from modules import shared, sd_models
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img


IS_WEBUI_1_9 = hasattr(shared.cmd_opts, 'unix_filenames_sanitization')
quote_swap = str.maketrans('\'"', '"\'')


def limiSizeByOneDemention(size: tuple, limit: int):
    w, h = size
    if h > w:
        if h > limit:
            w = limit / h * w
            h = limit
    else:
        if w > limit:
            h = limit / w * h
            w = limit

    return (int(w), int(h))


def getJobsCountTxt2Img(originalP: StableDiffusionProcessingTxt2Img) -> int:
    jobs = originalP.n_iter
    secondpass_count = originalP.batch_size * originalP.n_iter
    jobs += secondpass_count
    if originalP.enable_hr:
        jobs += secondpass_count
    return jobs


def getTotalStepsTxt2Img(originalP: StableDiffusionProcessingTxt2Img, firstpass_steps: int, firstpass_denoising: float) -> int:
    totalSteps = firstpass_steps * originalP.n_iter
    secondpass_count = originalP.batch_size * originalP.n_iter
    totalSteps += secondpass_count * min(math.ceil(originalP.steps * firstpass_denoising + 1), originalP.steps)
    if originalP.enable_hr:
        totalSteps += secondpass_count * originalP.hr_second_pass_steps
    return totalSteps


def getJobsCountImg2Img(originalP: StableDiffusionProcessingImg2Img) -> int:
    return 1 + originalP.n_iter


def getTotalStepsImg2Img(originalP: StableDiffusionProcessingImg2Img, firstpass_steps: int, firstpass_denoising: float) -> int:
    totalSteps = min(math.ceil(firstpass_steps * originalP.denoising_strength + 1), firstpass_steps)
    totalSteps += originalP.n_iter * min(math.ceil(originalP.steps * firstpass_denoising + 1), originalP.steps)
    return totalSteps

def getSecondPassBeginFromImg2Img(originalP: StableDiffusionProcessingImg2Img, firstpass_steps: int) -> int:
    totalSteps = min(math.ceil(firstpass_steps * originalP.denoising_strength + 1), firstpass_steps)
    return totalSteps


def convert_txt2img_to_img2img(txt2img: StableDiffusionProcessingTxt2Img) -> StableDiffusionProcessingImg2Img:
    txt2imgKWArgs = {}
    txt2imgArgs = ['sd_model', 'outpath_samples', 'outpath_grids', 'prompt', 'negative_prompt', 'styles',
        'sampler_name', 'batch_size', 'n_iter', 'steps', 'cfg_scale', 'width', 'height', 'override_settings',
        'do_not_save_samples', *(['scheduler'] if IS_WEBUI_1_9 else [])
    ]
    for arg in txt2imgArgs:
        txt2imgKWArgs[arg] = getattr(txt2img, arg, None)

    img2imgKWArgs = {
        'init_images': [],
        'mask': None,
        'mask_blur': 2,
        'inpainting_fill': 0,
        'resize_mode': 0,
        'denoising_strength': 0.5,
        'image_cfg_scale': 1.5,
        'inpaint_full_res': False,
        'inpaint_full_res_padding': 90,
        'inpainting_mask_invert': False,
    }

    img2img = StableDiffusionProcessingImg2Img(**txt2imgKWArgs, **img2imgKWArgs)

    otherArgs = ['seed', 'subseed', 'subseed_strength', 'refiner_checkpoint', 'refiner_checkpoint',
        'refiner_switch_at', 'seed_resize_from_h', 'seed_resize_from_w', 'extra_generation_params']

    for arg in otherArgs:
        value = getattr(txt2img, arg, None)
        setattr(img2img, arg, value)

    return img2img


def interrupted():
    return shared.state.interrupted or getattr(shared.state, 'stopping_generation', False)


def _removeAllNetworksWithErrorsWarnings(string: str) -> str:
    resLines = []
    for line in string.split('\n'):
        if not line.startswith('Networks with errors:'):
            resLines.append(line)
    return '\n'.join(resLines)

    

def removeAllNetworksWithErrorsWarnings(processed: Processed):
    processed.comments = _removeAllNetworksWithErrorsWarnings(processed.comments)

NAME = "Old SD firstpasser"


def get_model_short_title(model_aliases):
    if model := sd_models.get_closet_checkpoint_match(model_aliases):
        return model.short_title
    return model_aliases
