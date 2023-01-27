from transformers import pipeline
import torch
import whisper_timestamped as whisper

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Run the model
    audio = whisper.load_audio("https://delyrium.s3.eu-west-3.amazonaws.com/hond1.mp3")
    result = whisper.transcribe(model, audio)

    # Return the results as a dictionary
    return result
