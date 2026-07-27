# Check required Python packages and install any that are missing or out of range.
import gradio as gr
from transformers import pipeline
from transformers.utils import logging
import numpy as np

from helper import (
    ignore_warnings,
    render_results_in_image,
    summarize_predictions_natural_language,
)

# Suppress warning messages
logging.set_verbosity_error()
ignore_warnings()

from phonemizer.backend.espeak.wrapper import EspeakWrapper

EspeakWrapper.set_library(
    r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
)

# Import the model and create an instance
od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")

# other model
# Text to Speech model
tts_pipe = pipeline("text-to-speech", model="kakao-enterprise/vits-ljs")


# this function will do all the steps under the hood
# updating the pipelint to do the audio as well in the gradio app.
# detect, annotate, smmarize and generate audio summary.

def get_pipeline_prediction(pil_image):

    if pil_image is None:
        return None, "Please upload image!", None

    # this will process the image and identify the objects
    predictions = od_pipe(pil_image)

    # This will output the labelled image with boxes and confidence score
    processed_image = render_results_in_image(pil_image, predictions)

    #conver to labels , text.
    description = summarize_predictions_natural_language(predictions)



    try: 
        #convert audio
        narrated = tts_pipe(description)

        # *** gradio audio shape, 1d (1, ssamples)
        audio = np.asarray(narrated["audio"]).squeeze()
        sampling = narrated["sampling_rate"]

        audio_output = (sampling, audio)
    except Exception as error:
        print(f"Audio generation failed: {error}")
        audio_output = None

    return processed_image,description, audio_output
    
demo = gr.Interface(
  fn=get_pipeline_prediction,
  inputs=gr.Image(label="Input image",
                  type="pil"),
  outputs= [
              gr.Image(label="Output image with predicted instances", type="pil"),
              gr.Textbox(label="Image Description", lines = 3),
              gr.Audio(label = "Audio Description", autoplay = False)
  ],
  title = "ML2_8: Object Detection with Audio Assist",
  description = ("Upload an image to detect objects with bound boxes, "
                 "generate a descripion and playback in audio."
  )
)

# `share=True` will provide an online link to access to the demo'''
# share=True will create a public link for sharing

if __name__ == "__main__":
    demo.launch()