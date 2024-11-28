from cached_path import cached_path

from dp.phonemizer import Phonemizer
import gradio as gr
from styletts2 import tts
import torch
import numpy as np
from pathlib import Path
from txtsplit import txtsplit
import os
import pickle

# Initialize the TTS model
model_checkpoint_path = '/Models/LibriTTS/epochs_2nd_00020.pth'
config_path = '/Models/LibriTTS/config.yml'
my_tts = tts.StyleTTS2(model_checkpoint_path=model_checkpoint_path, config_path=config_path)

# global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))

# espeak backend phonemizer initialisation
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)


def clsynthesize(text, voice, vcsteps, embscale, alpha, beta, progress=gr.Progress()):
    """
    The function to synthesize speech based on text and voice style.
    It handles voice cloning and the synthesis process.
    """
    # Check for empty text or text exceeding character limit
    if text.strip() == "":
        raise gr.Error("You must enter some text")
    if len(text) > 50000:
        raise gr.Error("Text must be <50k characters")
    if embscale > 1.3 and len(text) < 20:
        gr.Warning("WARNING: Short text with high embedding scale might cause static!")

    print("*** saying ***")
    print(text)
    print("*** end ***")

    texts = txtsplit(text)
    audios = []

    # Check if the voice input is a valid path
    if isinstance(voice, str) and Path(voice).exists():
        vs = my_tts.compute_style(voice)
    else:
        raise gr.Error("Invalid voice input. Please provide a valid voice audio file.")

    # Generate audio for each chunk
    for t in progress.tqdm(texts):
        audios.append(
            my_tts.inference(t, ref_s=vs, alpha=alpha, beta=beta, diffusion_steps=vcsteps, embedding_scale=embscale))

    # Concatenate audio and return
    return (24000, np.concatenate(audios))


# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## StyleTTS2 - Text-to-Speech with Style Transfer")

    with gr.Blocks() as clone:
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
            clvoice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=1000, waveform_options={'waveform_progress_color': '#3C82F6'})
            vcsteps = gr.Slider(minimum=3, maximum=20, value=20, step=1, label="Diffusion Steps", info="Higher steps should provide better quality but can be slower.", interactive=True)
            embscale = gr.Slider(minimum=1, maximum=10, value=1, step=0.1, label="Embedding Scale (READ WARNING BELOW)", info="Defaults to 1. WARNING: If this is too high and you generate short text, it may cause static!", interactive=True)
            alpha = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.1, label="Alpha", info="Defaults to 0.3", interactive=True)
            beta = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.1, label="Beta", info="Defaults to 0.7", interactive=True)
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio", waveform_options={'waveform_progress_color': '#3C82F6'})
            clbtn.click(clsynthesize, inputs=[clinp, clvoice, vcsteps, embscale, alpha, beta], outputs=[claudio], concurrency_limit=4)

# Please do not remove this line.
if __name__ == "__main__":
    # Launch the Gradio interface
    demo.queue(api_open=False, max_size=15).launch(show_api=False)
