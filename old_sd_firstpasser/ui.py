import json
import gradio as gr
from modules import ui_settings, shared
from old_sd_firstpasser.tools import quote_swap, NAME


def makeUI(script):
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
    with gr.Row():
        sdxl_checkpoint = ui_settings.create_setting_component('sd_model_checkpoint')
        sdxl_checkpoint.label = "Checkpoint for SDXL pass"
    with gr.Row():
        network_type = gr.Radio(value="Auto", choices=["Auto", "SD1", "SDXL"],
            label="Firstpass network type", info="Auto means guess by loras metadata. "
            "For ControlNet and other networks set it up manually",
            elem_classes=['compact-checkbox-group'])
    def get_infotext_field(d, field):
        if NAME in d:
            return d[NAME].get(field)

    script.infotext_fields = [
        (firstpass_steps, lambda d: get_infotext_field(d, 'steps')),
        (firstpass_denoising, lambda d: get_infotext_field(d, 'denoising')),
        (firstpass_upscaler, lambda d: get_infotext_field(d, 'upscaler')),
        (sd_1_checkpoint, lambda d: get_infotext_field(d, 'model_sd1')),
        (sdxl_checkpoint, lambda d: get_infotext_field(d, 'model_sdxl')),
        (network_type, lambda d: get_infotext_field(d, 'network_type')),
    ]

    return [firstpass_steps, firstpass_denoising, firstpass_upscaler, sd_1_checkpoint,
            sdxl_checkpoint, network_type]


def pares_infotext(infotext, params):
    try:
        params[NAME] = json.loads(params[NAME].translate(quote_swap))
    except Exception:
        pass