import copy
from contextlib import closing
import gradio as gr
from modules import shared, scripts
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, process_images

from old_sd_firstpasser.tools import ( convert_txt2img_to_img2img, limiSizeByOneDemention,
    getJobsCountTxt2Img, getTotalStepsTxt2Img, interrupted, removeAllNetworksWithErrorsWarnings,
)
from old_sd_firstpasser.ui import makeUI


class Script(scripts.Script):
    def title(self):
        return "Old SD firstpasser"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        ui = makeUI()
        gr.Markdown("If you want to use extensions on second pass too "\
                "(e.g. sdxl controlnet in addition to sd 1.5), please use img2img "\
                "tab, and left initial image empty")
        return ui


    def run(self, originalP: StableDiffusionProcessingTxt2Img, firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint):
        oringinalCheckpoint = shared.opts.sd_model_checkpoint if not 'sd_model_checkpoint' in originalP.override_settings else originalP.override_settings['sd_model_checkpoint']
        if getattr(originalP, 'firstpass_image', False):
            return None
        shared.total_tqdm.clear()
        shared.state.job_count = getJobsCountTxt2Img(originalP)
        shared.total_tqdm.updateTotal(getTotalStepsTxt2Img(originalP, firstpass_steps, firstpass_denoising))
        originalP.do_not_save_grid = True

        txt2imgP = copy.copy(originalP)
        txt2imgP.enable_hr = False
        txt2imgP.width, txt2imgP.height = limiSizeByOneDemention((originalP.width, originalP.height), 512)
        txt2imgP.override_settings['sd_model_checkpoint'] = sd_1_checkpoint
        txt2imgP.steps = firstpass_steps

        with closing(txt2imgP):
            shared.state.textinfo = "firstpassing with sd 1.x"
            processed1: Processed = process_images(txt2imgP)
        # throwning away all extra images e.g. controlnet preprocessed
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

        batchPocessed: Processed = None
        shared.state.textinfo = "generation"
        for image in processed1.images:
            img2imgP = convert_txt2img_to_img2img(originalP)
            img2imgP.batch_size = 1
            img2imgP.n_iter = 1
            img2imgP.init_images = [image]
            img2imgP.denoising_strength = firstpass_denoising
            img2imgP.override_settings['sd_model_checkpoint'] = oringinalCheckpoint
            img2imgP.override_settings['upscaler_for_img2img'] = firstpass_upscaler

            with closing(img2imgP):
                processed2: Processed = process_images(img2imgP)
            if batchPocessed:
                n = len(processed2.all_seeds)
                batchPocessed.images += processed2.images[:n]
                batchPocessed.infotexts += processed2.infotexts[:n]
                batchPocessed.all_seeds += processed2.all_seeds
                batchPocessed.all_subseeds += processed2.all_subseeds
                batchPocessed.all_negative_prompts += processed2.all_negative_prompts
                batchPocessed.all_prompts += processed2.all_prompts
                batchPocessed.comments += processed2.comments[:n]
            else:
                batchPocessed = processed2

            if interrupted():
                return processedOnExit(batchPocessed)
        
        if originalP.enable_hr and hasattr(originalP, 'firstpass_image'):
            for i in range(len(batchPocessed.all_seeds)):
                shared.state.textinfo = "applying hires fix"
                hiresP = copy.copy(originalP)
                hiresP.firstpass_image = batchPocessed.images[i]
                hiresP.batch_size = 1
                hiresP.n_iter = 1
                hiresP.override_settings['save_images_before_highres_fix'] = False
                hiresP.txt2img_upscale = True # txt2img_upscale attribute that signifies this is called by txt2img_upscale
                hiresP.seed = batchPocessed.all_seeds[i]
                with closing(hiresP):
                    processedHR: Processed = process_images(hiresP)
                batchPocessed.images[i] = processedHR.images[0]
                if interrupted():
                    return processedOnExit(batchPocessed)

        return processedOnExit(batchPocessed)
