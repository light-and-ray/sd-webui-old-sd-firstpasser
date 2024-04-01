import copy
from contextlib import closing
import gradio as gr
from PIL import Image
from modules import ui_settings, shared, scripts_postprocessing, scripts, sd_models, processing
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images

from old_sd_firstpasser.tools import ( convert_txt2img_to_img2img, limiSizeByOneDemention,
    getJobsCountImg2Img, getTotalStepsImg2Img, interrupted, removeAllNetworksWithErrorsWarnings,
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


NAME = "Old SD firstpasser"

class Script(scripts.Script):
    def __init__(self):
        self.enable = None
        self.scriptsImages = []
        self.scriptsInfotexts = []

    def title(self):
        return NAME

    def show(self, is_img2img):
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        with (
            InputAccordion(False, label=NAME) if InputAccordion
            else gr.Accordion(NAME, open=False)
            as enable
        ):
            if not InputAccordion:
                with gr.Row():
                    enable = gr.Checkbox(False, label="Enable")
            ui = makeUI()
        return [enable] + ui


    def before_process(self, originalP: StableDiffusionProcessingImg2Img, enable, firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint):
        if getattr(originalP, 'old_sd_firstpasser_prevent_recursion', False):
            return

        self.enable = enable
        if not self.enable:
            if getattr(originalP.init_images[0], 'inited_by_old_sd_firstpasser', False):
                originalP.init_images[0] = None
                raise Exception("No input image")
            return

        oringinalCheckpoint = shared.opts.sd_model_checkpoint if not 'sd_model_checkpoint' in originalP.override_settings else originalP.override_settings['sd_model_checkpoint']

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
            shared.total_tqdm.updateTotal(getTotalStepsImg2Img(originalP, firstpass_steps, firstpass_denoising))

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
            originalP.override_settings['upscaler_for_img2img'] = firstpass_upscaler
        finally:
            shared.state.textinfo = "switching sd checkpoint"
            shared.opts.sd_model_checkpoint = oringinalCheckpoint
            sd_models.reload_model_weights()
            shared.state.textinfo = "generating"


    def postprocess(self, originalP: StableDiffusionProcessingImg2Img, processed: Processed, *args):
        if not self.enable or getattr(originalP, 'old_sd_firstpasser_prevent_recursion', False):
            return
        processed.images += self.scriptsImages
        processed.infotexts += self.scriptsInfotexts
        removeAllNetworksWithErrorsWarnings(processed)
