import torch
import requests
import whisper

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)

def callback(result):
    url = 'https://scribewave.com/api/whisper/banana/callback'
    headers = {'Content-type': 'application/json'}
    data = {
        "projectId": "random",
        "transcribeResult": result,
        "secret": "curieuze5neuze8mosterdpot33"
    }
    response = requests.post(url, json=data, headers=headers)
    print(response.json())


# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs: dict) -> dict:
    global model

    # Parse out your arguments
    url = model_inputs.get('url', None)
    if url == None:
        return {'message': "No url provided"}

    # Run the model
    audio = whisper.load_audio(url)
    result = whisper.transcribe(model, audio)

    # Callback
    callback(result)

    # Return the results as a dictionary
    return result