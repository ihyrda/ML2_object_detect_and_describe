# Gradio Object Detection with Audio Assistance

## Project description

This application uses the pretrained Hugging Face facebook/detr-resnet model to detect objects in uploaded images, and annotate text and audio assist.


## Original project

This lab was adapted from:
altafumer/Object_detection_using_HF

## Models

- Object detection: facebook/detr-resnet-50
- Text-to-speech: kakao-enterprise/vits-ljs

## Installation

1. Clone this repository.
2. Create or activate a Python environment.
3. Install the Python dependencies:

   pip install -r requirements.txt

4. Install eSpeak NG for audio narration.  
*** the script  is hardcoded to local path default that may not be identical for your system. ***

## Running the application

python app.py

Then open the local Gradio address shown in the terminal.

## Files

- `app.py`: standalone Gradio application
- `helper.py`: prediction rendering and summary functions
- `notebook/`: completed instructional notebook
- `test_images/`: the three test inputs
- `results/`: screenshots or saved outputs

## Limitations

The detector uses a pretrained model and may miss small, partially hidden, or unsupported objects. 
Audio generation can take longer when running locally on a CPU.