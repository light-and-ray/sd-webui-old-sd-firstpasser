import math
from modules import shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img


IS_WEBUI_1_9 = hasattr(shared.cmd_opts, 'unix_filenames_sanitization')


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


def getJobsCount(originalP: StableDiffusionProcessingTxt2Img) -> int:
    jobs = originalP.n_iter
    secondpass_count = originalP.batch_size * originalP.n_iter
    jobs += secondpass_count
    if originalP.enable_hr:
        jobs += secondpass_count
    return jobs


def getTotalSteps(originalP: StableDiffusionProcessingTxt2Img, firstpass_steps: int, firstpass_denoising: float) -> int:
    totalSteps = firstpass_steps * originalP.n_iter
    secondpass_count = originalP.batch_size * originalP.n_iter
    totalSteps += secondpass_count * min(math.ceil(originalP.steps * firstpass_denoising + 1), originalP.steps)
    if originalP.enable_hr:
        totalSteps += secondpass_count * originalP.hr_second_pass_steps
    return totalSteps


def convert_txt2img_to_img2img(txt2img: StableDiffusionProcessingTxt2Img) -> StableDiffusionProcessingImg2Img:
    txt2imgKWArgs = {}
    txt2imgArgs = ['sd_model', 'outpath_samples', 'outpath_grids', 'prompt', 'negative_prompt', 'styles',
        'sampler_name', 'batch_size', 'n_iter', 'steps', 'cfg_scale', 'width', 'height', 'override_settings',
        'do_not_save_samples', *(['scheduler'] if IS_WEBUI_1_9 else [])
    ]
    for arg in txt2imgArgs:
        txt2imgKWArgs[arg] = getattr(txt2img, arg)

    img2imgArgs = {
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

    img2img = StableDiffusionProcessingImg2Img(**txt2imgKWArgs, **img2imgArgs)

    otherArgs = ['seed', 'subseed', 'subseed_strength', 'refiner_checkpoint', 'refiner_checkpoint',
        'refiner_switch_at', 'seed_resize_from_h', 'seed_resize_from_w']

    for arg in otherArgs:
        value = getattr(txt2img, arg)
        setattr(img2img, arg, value)

    # it looks like enabling txt2img scripts in img2img is hard

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
