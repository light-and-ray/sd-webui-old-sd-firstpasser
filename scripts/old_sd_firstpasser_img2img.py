import copy
from contextlib import closing
import gradio as gr
from PIL import Image
from modules import shared, scripts_postprocessing, scripts, sd_models, processing
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images

from old_sd_firstpasser.tools import ( limiSizeByOneDemention, getJobsCountImg2Img,
    getTotalStepsImg2Img, removeAllNetworksWithErrorsWarnings, NAME, getSecondPassBeginFromImg2Img,
)
from old_sd_firstpasser.ui import makeUI
if hasattr(scripts_postprocessing.ScriptPostprocessing, 'process_firstpass'):  # webui >= 1.7
    from modules.ui_components import InputAccordion
else:
    InputAccordion = None


original_StableDiffusionProcessingImg2Img__init__ = processing.StableDiffusionProcessingImg2Img.__init__

def hijack_fill_dummy_init_images(*args, **kwargs):
    if not kwargs["init_images"] or not all(kwargs["init_images"]):
        dummy_image = Image.new('RGB', (kwargs['width'], kwargs['height']))
        dummy_image.inited_by_old_sd_firstpasser = True
        kwargs["init_images"] = [dummy_image]

    return original_StableDiffusionProcessingImg2Img__init__(*args, **kwargs)

processing.StableDiffusionProcessingImg2Img.__init__ = hijack_fill_dummy_init_images



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
        ui = makeUI()
        return ui


    def run(self, originalP: StableDiffusionProcessingImg2Img, firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint):
        oringinalCheckpoint = shared.opts.sd_model_checkpoint if not 'sd_model_checkpoint' in originalP.override_settings else originalP.override_settings['sd_model_checkpoint']
        self.originalUpscaler = shared.opts.upscaler_for_img2img
        try:
            shared.state.textinfo = "switching sd checkpoint"
            shared.opts.sd_model_checkpoint = sd_1_checkpoint
            sd_models.reload_model_weights()

            originalP.do_not_save_grid = True

            img2imgP = copy.copy(originalP)
            img2imgP.width, img2imgP.height = limiSizeByOneDemention((originalP.width, originalP.height), 512)
            img2imgP.steps = firstpass_steps
            img2imgP.batch_size = 1
            img2imgP.n_iter = 1
    
            if getattr(originalP.init_images[0], 'inited_by_old_sd_firstpasser', False): # txt2img equivalent
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
                shared.state.textinfo = "firstpassing with sd 1.x"
                processed1: Processed = process_images(img2imgP)
            # throwning away all extra images e.g. controlnet preprocessed
            n = len(processed1.all_seeds)
            self.scriptsImages = processed1.images[n:]
            self.scriptsInfotexts = processed1.infotexts[n:]
            originalP.init_images = processed1.images[:n]
            originalP.denoising_strength = firstpass_denoising
        finally:
            shared.state.textinfo = "switching sd checkpoint"
            shared.opts.sd_model_checkpoint = oringinalCheckpoint
            sd_models.reload_model_weights()
            shared.state.textinfo = "generating"
            self.firstpass_upscaler = firstpass_upscaler
            originalP.selectable_old_sd_firstpasser_script = self




class ScriptBackground(scripts.Script):
    def title(self):
        return NAME

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
