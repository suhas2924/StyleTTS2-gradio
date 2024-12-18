from cached_path import cached_path

from dp.phonemizer import Phonemizer
import gradio as gr
from styletts2 import tts
import torch
import numpy as np
from pathlib import Path
from txtsplit import txtsplit
import os
import re

theme = gr.themes.Ocean(
    font=[gr.themes.GoogleFont('Merryweather'), gr.themes.GoogleFont('Public Sans'), 'system-ui', 'sans-serif'],
)

# Initialize the TTS model
model_checkpoint_path = '/Models/LibriTTS/epochs_2nd_00020.pth'
config_path = '/Models/LibriTTS/config.yml'
my_tts = tts.StyleTTS2(model_checkpoint_path=model_checkpoint_path, config_path=config_path)

# global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
phonemizer = Phonemizer.from_checkpoint(str(cached_path('https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt')))

# espeak backend phonemizer initialisation
import phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)

def preprocess_to_ignore_quotes(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+', ' ', text)  # Collapsing multiple spaces/tabs into one
    return text
    
# Split the text into paragraphs and handle spacing
def segment_text(text):
    """
    Splits the text into paragraphs while keeping the paragraphs as separate chunks.
    Handles extra spaces and newline characters properly, while keeping paragraphs intact.
    """
    # Preprocess the text (remove quotes and handle unwanted characters)
    cleaned_text = preprocess_to_ignore_quotes(text)
    
    # Split the cleaned text into paragraphs based on one or more newline characters
    # Adjust splitting logic to handle both newlines and blank lines
    text_segments = re.split(r'(?:\n\s*\n|\n{2,})', cleaned_text)
    
    # Return paragraphs with extra spaces removed within each paragraph
    return [text_segment.strip() for text_segment in text_segments if text_segment.strip()]

def clsynthesize(text, 
                 sample_rate, 
                 voice, 
                 vcsteps, 
                 embscale, 
                 alpha, 
                 beta,
                 t,
                 phonemize=False,
                 progress=gr.Progress()):
    """
    The function to synthesize speech based on text and voice style.
    It handles voice cloning and the synthesis process.
    """
    
    text_segments = segment_text(text)
    # Debugging step: print the paragraphs
    print(f"text_segments: {text_segments}")
    
    audios = []
    sr = sample_rate
    vs = my_tts.compute_style(voice)
    prev_s = None

    # Apply phonemizer if needed
    if phonemize:
        # Use the phonemizer to convert the text into phonetic representation (IPA)
        text_segments = [global_phonemizer.phonemize([text_segment])[0] for text_segment in text_segments]

    # Generate audio for each chunk
    for text_segment in progress.tqdm(text_segments):
        
        audio, prev_s = my_tts.long_inference_segment(text_segment, prev_s, ref_s=vs, alpha=alpha, beta=beta, t=t, diffusion_steps=vcsteps, embedding_scale=embscale)
        audios.append(audio)

    # Concatenate audio and return
    return (sr, np.concatenate(audios))


# Gradio Interface
with gr.Blocks(theme=theme) as demo:
    gr.Markdown("## StyleTTS2 - Text-to-Speech")

    with gr.Blocks() as clone:
        with gr.Column(scale=1):
            clinp = gr.Textbox(label="Text", info="What would you like StyleTTS 2 to read? It works better on full sentences.", interactive=True)
        with gr.Column(scale=1):
            voice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=1000, waveform_options={'waveform_progress_color': '#3C82F6'})
        with gr.Column(scale=1):
            clsample = gr.Slider(minimum=16000, maximum=48000, value=24000, step=50, label="Sample Rate", interactive=True)
            vcsteps = gr.Slider(minimum=3, maximum=20, value=20, step=1, label="Diffusion Steps", info="Higher steps should provide better quality but can be slower.", interactive=True)
            embscale = gr.Slider(minimum=0.5, maximum=10, value=1, step=0.1, label="Embedding Scale (READ WARNING BELOW)", info="Defaults to 0. WARNING: If this is too high and you generate short text, it may cause static!", interactive=True)
            alpha = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.1, label="Alpha", info="Defaults to 0.3", interactive=True)
            beta = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.1, label="Beta", info="Defaults to 0.7", interactive=True)
            t = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, label="consistency", interactive=True)
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio", waveform_options={'waveform_progress_color': '#3C82F6'})
            clbtn.click(clsynthesize, inputs=[clinp, clsample, voice, vcsteps, embscale, alpha, beta, t], outputs=[claudio], concurrency_limit=1)

    # Launch the Gradio interface
    demo.queue(api_open=False, max_size=1).launch(show_api=False)
