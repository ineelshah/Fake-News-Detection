import os
from PIL import Image

def calculate_resolution(filename):
    image = Image.open(filename)
    width, height = image.size
    return width, height

