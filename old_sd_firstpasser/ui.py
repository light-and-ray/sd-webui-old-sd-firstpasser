import copy
from contextlib import closing
import gradio as gr
import modules.scripts as scripts
from modules import ui_settings, shared
from modules.processing import Processed, StableDiffusionProcessingTxt2Img, process_images



def makeUI():
    with gr.Row():
        firstpass_steps = gr.Slider(
            label='Firstpass steps',
            value=20,
            step=1,
            minimum=1,
            maximum=150,
            elem_id="firstpass_steps"
        )
        firstpass_denoising = gr.Slider(label='Firstpass denoising',
            value=0.55, elem_id="firstpass_denoising",
            minimum=0.0, maximum=1.0, step=0.01
        )
    with gr.Row():
        firstpass_upscaler = gr.Dropdown(
            value="ESRGAN_4x",
            choices=[x.name for x in shared.sd_upscalers],
            label="Firstpass upscaler",
            elem_id="firstpass_upscaler",
        )
    with gr.Row():
        sd_1_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')
        sd_1_checkpoint.label = "Checkpoint for SD 1.x pass"

    return [firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint]

