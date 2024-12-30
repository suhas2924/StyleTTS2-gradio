from cached_path import cached_path

import sys

import gradio as gr
from styletts2 import tts
import numpy as np
from pathlib import Path
import torch

from phonemizer.backend import EspeakBackend
from phonemizer.backend.base import BaseBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.punctuation import Punctuation
from phonemizer import phonemize

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

# espeak backend phonemizer initialisation
import phonemizer

global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', punctuation_marks=Punctuation.default_marks(), preserve_punctuation=True, with_stress=True)
    
SPLIT_WORDS = [
    "but", "then", "so", "however", "nevertheless", "yet", "still", "accordingly", "consequently",
    "therefore", "thus", "hence", "consequently", "after", "subsequently",
    "moreover", "furthermore", "additionally", "nonetheless", "also", "besides",
    "meanwhile", "alternatively", "otherwise", "nevertheless", "meanwhile",
    "namely", "specifically", "for example", "such as", "and",
    "in fact", "indeed", "notably", "instead", "likewise",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]

def preprocess_to_ignore_quotes(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[“”"]', '', text)  # Remove both fancy quotes and normal quotes
    # Temporarily replace existing ellipses (...) with a placeholder
    text = re.sub(r'\.\.\.|\. \. \.|…', '###ELLIPSIS###', text)
    text = re.sub(r'[.]', '...', text)
    # Restore the placeholder back to actual ellipses (...)
    text = re.sub(r'###ELLIPSIS###', '...', text)
     # Normalize uppercase words to title case unless they are acronyms
    text = re.sub(r'\b([A-Z]{2,})\b', lambda x: x.group(0).capitalize(), text)
    text = re.sub(r'[ \t]+', ' ', text)  # Collapsing multiple spaces/tabs into one
    print ("Cleaned Text", text)
    return text
    
    
def segment_text(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if not text or text[-1] not in ['。', '...']:
        text += '...'

    sentences = re.split(r'(\.\.\.)', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]
    
    batches = []
    current_batch = ""

    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""
            
            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence

    if current_batch:
        batches.append(current_batch)

    return batches
    
def clsynthesize(sample_rate, 
                 voice, 
                 vcsteps, 
                 embscale, 
                 alpha, 
                 beta,
                 t,
                 text=None,
                 uploaded_file=None,
                 use_gruut=False,
                 progress=gr.Progress()):
    """
    The function to synthesize speech based on text and voice style.
    It handles voice cloning and the synthesis process.
    """

    # Ensure either text or a file is provided
    if not text and not uploaded_file:
        return "Please provide text or upload a file."

    # If a file is uploaded, read its contents
    if uploaded_file is not None:
        try:
            with open(uploaded_file, 'r', encoding='utf-8') as file:
                file_text = file.read()
        except Exception as e:
            return f"Error reading uploaded file: {e}"

        # If both text and file are provided, concatenate them
        if text:
            text = f"{text}\n{file_text}"
        else:
            text = file_text
            
    # Preprocess the text (e.g., clean up quotes and spaces)
    text = preprocess_to_ignore_quotes(text)
    
    text_segments = segment_text(text)
    # Debugging step: print the paragraphs
    print(f"text_segments: {text_segments}")
    
    audios = []
    sr = sample_rate
    vs = my_tts.compute_style(voice)
    prev_s = None

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
            voice = gr.Audio(label="Voice", interactive=True, type='filepath', max_length=1000, waveform_options={'waveform_progress_color': '#3C82F6'})
        with gr.Column(scale=1):
            clsample = gr.Slider(minimum=16000, maximum=48000, value=24000, step=50, label="Sample Rate", interactive=True)
            vcsteps = gr.Slider(minimum=3, maximum=20, value=20, step=1, label="Diffusion Steps", info="Higher steps should provide better quality but can be slower.", interactive=True)
            embscale = gr.Slider(minimum=1, maximum=10, value=1, step=0.1, label="Embedding Scale (READ WARNING BELOW)", info="Defaults to 0. WARNING: If this is too high and you generate short text, it may cause static!", interactive=True)
            alpha = gr.Slider(minimum=0, maximum=1, value=0.3, step=0.1, label="Alpha", info="Defaults to 0.3", interactive=True)
            beta = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.1, label="Beta", info="Defaults to 0.7", interactive=True)
            t = gr.Slider(minimum=0, maximum=1, value=0.7, step=0.01, label="consistency", interactive=True)
        with gr.Column(scale=1):
            clinp = gr.Textbox(
                label="Input Text",
                interactive=True,
                placeholder="Enter text : what would you like me to read?",
                show_label=True,
            )
            upload_file_clone = gr.File(
                label="Upload Text File", type="filepath", interactive=True
            )
        with gr.Column(scale=1):
            clbtn = gr.Button("Synthesize", variant="primary")
            claudio = gr.Audio(interactive=False, label="Synthesized Audio", waveform_options={'waveform_progress_color': '#3C82F6'})
            clbtn.click(clsynthesize, inputs=[clsample, voice, vcsteps, embscale, alpha, beta, t, clinp, upload_file_clone], outputs=[claudio], concurrency_limit=1)
        
    # Launch the Gradio interface
    demo.queue(api_open=False, max_size=1).launch(show_api=False)
