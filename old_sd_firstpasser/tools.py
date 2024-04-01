import math
from modules import shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, StableDiffusionProcessingImg2Img
from modules.shared import opts, state



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
    totalSteps += math.ceil(secondpass_count * originalP.steps * firstpass_denoising)
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
    # img2img.scripts = copy.copy(txt2img.scripts)
    # img2img.scripts.initialize_scripts(is_img2img=True)
    # img2img.script_args = txt2img.script_args
    return img2img


def interrupted():
    return shared.state.interrupted or getattr(shared.state, 'stopping_generation', False)


# def _removeAllNetworksWithErrorsWarnings(string: str) -> str:
#     resLines = []
#     for line in string.split('\n'):
#         if not line.startswith('Networks with errors:'):
#             resLines.append(line)
#     return '\n'.join(resLines)

    

# def removeAllNetworksWithErrorsWarnings(processed: Processed):
#     for i in range(len(processed.infotexts)):
#         processed.infotexts[i] = _removeAllNetworksWithErrorsWarnings(processed.infotexts[i])
#     processed.info = _removeAllNetworksWithErrorsWarnings(processed.info)
