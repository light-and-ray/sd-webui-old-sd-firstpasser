import copy
import json
from contextlib import closing
import gradio as gr
from modules import shared, scripts, script_callbacks
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, process_images

from old_sd_firstpasser.tools import ( convert_txt2img_to_img2img, limitSizeByOneDimension,
    getJobsCountTxt2Img, getTotalStepsTxt2Img, interrupted, removeAllNetworksWithErrorsWarnings,
    NAME, quote_swap, get_model_short_title,
)
from old_sd_firstpasser.ui import makeUI, pares_infotext


class Script(scripts.Script):
    def title(self):
        return NAME

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        ui = makeUI(self)
        gr.Markdown("If you want to use extensions on second pass too "\
                "(e.g. sdxl controlnet in addition to sd 1.5), please use img2img "\
                "tab, and left initial image empty")
        return ui


    def run(self, originalP: StableDiffusionProcessingTxt2Img, firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint):
        originalCheckpoint = shared.opts.sd_model_checkpoint if not 'sd_model_checkpoint' in originalP.override_settings else originalP.override_settings['sd_model_checkpoint']
        if getattr(originalP, 'firstpass_image', False):
            return None
        shared.total_tqdm.clear()
        shared.state.job_count = getJobsCountTxt2Img(originalP)
        shared.total_tqdm.updateTotal(getTotalStepsTxt2Img(originalP, firstpass_steps, firstpass_denoising))
        originalP.do_not_save_grid = True

        originalP.extra_generation_params['Script'] = NAME
        originalP.extra_generation_params[NAME] = json.dumps({
                'steps': firstpass_steps,
                'denoising': firstpass_denoising,
                'upscaler': firstpass_upscaler,
                'model': get_model_short_title(sd_1_checkpoint),
        }).translate(quote_swap)

        txt2imgP = copy.copy(originalP)
        txt2imgP.enable_hr = False
        txt2imgP.width, txt2imgP.height = limitSizeByOneDimension((originalP.width, originalP.height), 512)
        txt2imgP.override_settings['sd_model_checkpoint'] = sd_1_checkpoint
        txt2imgP.override_settings['sd_vae'] = 'Automatic'
        txt2imgP.steps = firstpass_steps

        with closing(txt2imgP):
            shared.state.textinfo = "firstpassing with sd 1.x"
            processed1: Processed = process_images(txt2imgP)
        # throning away all extra images e.g. controlnet preprocessed
        n = len(processed1.all_seeds)
        scriptsImages = processed1.images[n:]
        scriptsInfotexts = processed1.infotexts[n:]
        def processedOnExit(processed):
            processed.images += scriptsImages
            processed.infotexts += scriptsInfotexts
            removeAllNetworksWithErrorsWarnings(processed)
            return processed
        processed1.images = processed1.images[:n]
        processed1.infotexts = processed1.infotexts[:n]
        if interrupted():
            return processedOnExit(processed1)

        batchProcessed: Processed = None
        shared.state.textinfo = "generation"
        for idx in range(len(processed1.images)):
            img2imgP = convert_txt2img_to_img2img(originalP)
            img2imgP.batch_size = 1
            img2imgP.n_iter = 1
            img2imgP.init_images = [processed1.images[idx]]
            img2imgP.denoising_strength = firstpass_denoising
            img2imgP.override_settings['sd_model_checkpoint'] = originalCheckpoint
            img2imgP.override_settings['upscaler_for_img2img'] = firstpass_upscaler
            img2imgP.seed = processed1.all_seeds[idx]
            img2imgP.subseed = processed1.all_subseeds[idx]

            with closing(img2imgP):
                processed2: Processed = process_images(img2imgP)
            if batchProcessed:
                n = len(processed2.all_seeds)
                batchProcessed.images += processed2.images[:n]
                batchProcessed.infotexts += processed2.infotexts[:n]
                batchProcessed.all_seeds += processed2.all_seeds
                batchProcessed.all_subseeds += processed2.all_subseeds
                batchProcessed.all_negative_prompts += processed2.all_negative_prompts
                batchProcessed.all_prompts += processed2.all_prompts
                batchProcessed.comments += processed2.comments[:n]
            else:
                batchProcessed = processed2

            if interrupted():
                return processedOnExit(batchProcessed)

        if originalP.enable_hr and hasattr(originalP, 'firstpass_image'):
            for i in range(len(batchProcessed.all_seeds)):
                shared.state.textinfo = "applying hires fix"
                hiresP = copy.copy(originalP)
                hiresP.firstpass_image = batchProcessed.images[i]
                hiresP.batch_size = 1
                hiresP.n_iter = 1
                hiresP.override_settings['save_images_before_highres_fix'] = False
                hiresP.txt2img_upscale = True # txt2img_upscale attribute that signifies this is called by txt2img_upscale
                hiresP.seed = batchProcessed.all_seeds[i]
                with closing(hiresP):
                    processedHR: Processed = process_images(hiresP)
                batchProcessed.images[i] = processedHR.images[0]
                if interrupted():
                    return processedOnExit(batchProcessed)

        return processedOnExit(batchProcessed)

script_callbacks.on_infotext_pasted(pares_infotext)
