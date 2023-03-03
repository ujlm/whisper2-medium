from transformers import pipeline
import torch
import whisper_timestamped as whisper
import json
import requests

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
    audio = whisper.load_audio(prompt["url"])
    result = whisper.transcribe(model, audio)

    # HTTP callback to api to confirm result
    response = callback(result)

    # Return the results as a dictionary
    return result


def callback(result: dict):
    # HTTP callback to api to confirm result
    # define the URL to send the request to
    url = "http://localhost:3000/api/testEmail"

    # convert the data to JSON format
    json_result = json.dumps(result)

    # define the headers for the request
    headers = {"Content-Type": "application/json"}

    # send the POST request with the JSON payload
    response = requests.post(url, data=json_result, headers=headers)
    
    return response