import os
import torch
from model import S3D
from utils.load_weigths import load_model_weights

def main():
    n_classes = 400
    model = S3D(n_classes)
    model = load_model_weights(model, 'S3D_kinetics400.pt')
    
