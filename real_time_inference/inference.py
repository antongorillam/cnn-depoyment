import boto3
import torch
import io
import json
import os
import tarfile
import torchvision.transforms as transforms
from PIL import Image
import sys
from cnn import CNN 
from mnist import MNIST


# define any preprocessing transforms for your data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
def model_fn(model_dir):
    """Load the PyTorch model from the `model.tar.gz` file."""
    print("Loading model.")
    model = CNN()
    # extract model file from the archive
    print(f"os.listdir(): {os.listdir()}")
    print(f"os.listdir('code'): {os.listdir('code')}")
    print(f"os.path.join: {os.path.join(model_dir, f'model.pt')}")
    print(f"model_dir: {model_dir}")
    print(f"os.getcwd(): {os.getcwd()}")
    
    with open(os.path.join(model_dir, f'model.pt'), 'rb') as f:
        state = torch.load(f, map_location=torch.device('cpu'))['model_state_dict']
        model.load_state_dict(state)
        
    return model

def byte_to_tensor(image_bytes):
    """Preprocess the input image."""
    buffer = io.BytesIO(image_bytes)
    tensor = torch.load(buffer)
    return tensor

def input_fn(request_body, request_content_type):
    """Deserialize the input data and preprocess."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)['inputs']
        data = torch.tensor(data, dtype=torch.float32, device=device)
        return data
    else:
        raise ValueError(f'Unsupported content type: {request_content_type}')

def predict_fn(input_data, model):
    """Perform a forward pass of the model on the input data."""
    with torch.no_grad():
        output = model(input_data)
        return output

def output_fn(output_data, response_content_type):
    """Serialize the model output."""
    if response_content_type == 'application/json':
        result = output_data.cpu().numpy().tolist()
        return result
    else:
        raise ValueError(f'Unsupported content type: {response_content_type}')



"""

from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
IMG_DIM = 28
if __name__ == "__main__":
    model_dir = '.'
    model = model_fn(model_dir)

    mndata = MNIST('../mnist')
    test_images, test_labels = mndata.load_testing()
    test_images, test_labels = np.array(test_images)[0].reshape([1,1,28,28]), np.array(test_labels)[0]
    #test_images = torch.from_numpy(test_images).type(torch.FloatTensor).
    
    js = json.dumps({'inputs' : test_images.tolist()})
    parsed_input = input_fn(
        js, 
        request_content_type='application/json')
    
    pred = predict_fn(parsed_input, model)
    
    output = output_fn(
        pred, 
        response_content_type='application/json')
"""