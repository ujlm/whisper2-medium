import whisper
import torch

def download_model():
    model = whisper.load_model("medium")
if __name__ == "__main__":
    download_model()