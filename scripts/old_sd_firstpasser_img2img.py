import copy
from contextlib import closing
import json
from PIL import Image
from modules import shared, scripts_postprocessing, scripts, sd_models
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images

from old_sd_firstpasser.tools import (
    limitSizeByOneDimension, getJobsCountImg2Img, getTotalStepsImg2Img, removeAllNetworksWithErrorsWarnings, NAME,
    getSecondPassBeginFromImg2Img, quote_swap, get_model_short_title, guessNetworkType,
)
from old_sd_firstpasser.ui import makeUI
if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None



class ScriptSelectable(scripts.Script):
    def __init__(self):
        self.scriptsImages = []
        self.scriptsInfotexts = []
        self.originalUpscaler = None
        self.firstpass_upscaler = None
        self.total_tqdm_total = None
        self.total_tqdm_second_pass_begin_from = 0

    def title(self):
        return NAME

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        ui = makeUI(self)
        return ui


    def run(self, originalP: StableDiffusionProcessingImg2Img, firstpass_steps, firstpass_denoising,
            firstpass_upscaler, sd_1_checkpoint, sdxl_checkpoint, network_type):
        if network_type == "Auto":
            network_type = guessNetworkType(originalP)
        originalCheckpoint = shared.opts.sd_model_checkpoint if not 'sd_model_checkpoint' in originalP.override_settings else originalP.override_settings['sd_model_checkpoint']
        self.originalUpscaler = shared.opts.upscaler_for_img2img
        try:
            shared.state.textinfo = "switching sd checkpoint"
            if network_type == "SD1":
                shared.opts.sd_model_checkpoint = sd_1_checkpoint
            else: # SDXL
                shared.opts.sd_model_checkpoint = sdxl_checkpoint
            sd_models.reload_model_weights()

            originalP.do_not_save_grid = True

            originalP.extra_generation_params['Script'] = NAME
            originalP.extra_generation_params[NAME] = json.dumps({
                'steps': firstpass_steps,
                'denoising': firstpass_denoising,
                'upscaler': firstpass_upscaler,
                'model_sd1': get_model_short_title(sd_1_checkpoint),
                'model_sdxl': get_model_short_title(sdxl_checkpoint),
            }).translate(quote_swap)

            img2imgP = copy.copy(originalP)
            if network_type == 'SD1':
                img2imgP.width, img2imgP.height = limitSizeByOneDimension((originalP.width, originalP.height), 512)
            else: # SDXL
                img2imgP.width, img2imgP.height = limitSizeByOneDimension((originalP.width, originalP.height), 1024)
            img2imgP.steps = firstpass_steps
            img2imgP.batch_size = 1
            img2imgP.n_iter = 1
            img2imgP.override_settings['sd_vae'] = 'Automatic'

            if not originalP.init_images or not all(originalP.init_images): # txt2img equivalent
                dummy_image = Image.new('RGB', (originalP.width, originalP.height))
                img2imgP.init_images = [dummy_image]
                img2imgP.image_mask = Image.new('L', (img2imgP.width, img2imgP.height), 255)
                img2imgP.inpaint_full_res = False
                img2imgP.inpainting_fill = 2 # latent noise
                img2imgP.denoising_strength = 1.0
                originalP.denoising_strength = 1.0
            shared.state.job_count = getJobsCountImg2Img(originalP)
            self.total_tqdm_total = getTotalStepsImg2Img(originalP, firstpass_steps, firstpass_denoising)
            self.total_tqdm_second_pass_begin_from = getSecondPassBeginFromImg2Img(originalP, firstpass_steps)
            shared.total_tqdm.updateTotal(self.total_tqdm_total)

            with closing(img2imgP):
                img2imgP.old_sd_firstpasser_prevent_recursion = True
                shared.state.textinfo = f"firstpassing with {network_type.lower()}"
                processed1: Processed = process_images(img2imgP)
            # throwing away all extra images e.g. controlnet preprocessed
            n = len(processed1.all_seeds)
            self.scriptsImages = processed1.images[n:]
            self.scriptsInfotexts = processed1.infotexts[n:]
            originalP.init_images = processed1.images[:n]
            originalP.denoising_strength = firstpass_denoising
            originalP.seed = processed1.all_seeds[0]
            originalP.subseed = processed1.all_subseeds[0]
        finally:
            shared.state.textinfo = "switching sd checkpoint"
            shared.opts.sd_model_checkpoint = originalCheckpoint
            sd_models.reload_model_weights()
            shared.state.textinfo = "generating"
            self.firstpass_upscaler = firstpass_upscaler
            originalP.selectable_old_sd_firstpasser_script = self




class ScriptBackground(scripts.Script):
    def title(self):
        return NAME + " background"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        return []

    def before_process(self, originalP: StableDiffusionProcessingImg2Img, *args):
        selectable: ScriptSelectable = getattr(originalP, 'selectable_old_sd_firstpasser_script', None)
        if selectable is None:
            return

        if 'upscaler_for_img2img' in originalP.override_settings:
            del originalP.override_settings['upscaler_for_img2img']
        shared.opts.upscaler_for_img2img = selectable.firstpass_upscaler

        shared.total_tqdm.updateTotal(selectable.total_tqdm_total)
        for _ in range(selectable.total_tqdm_second_pass_begin_from):
            shared.total_tqdm.update()


    def postprocess(self, originalP: StableDiffusionProcessingImg2Img, processed: Processed, *args):
        selectable: ScriptSelectable = getattr(originalP, 'selectable_old_sd_firstpasser_script', None)
        if selectable is None:
            return
        processed.images += selectable.scriptsImages
        processed.infotexts += selectable.scriptsInfotexts
        removeAllNetworksWithErrorsWarnings(processed)
        if selectable.originalUpscaler:
            shared.opts.upscaler_for_img2img = selectable.originalUpscaler
