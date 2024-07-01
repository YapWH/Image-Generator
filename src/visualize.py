import numpy as np
import gradio as gr
import random
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle
import logging
from tqdm import tqdm
from kd import test

a = random.randint(0, 9)
b = random.randint(0, 9)
c = random.randint(0, 9)
d = random.randint(0, 9)

################################################################################################

test()

################################################################################################



################################################################################################

def Test(A):
	if A == f"{a,b,c,d}":
		img_dir = 'E:\AML\Image-Generator\src\results'
		imgs_List=[ os.path.join(img_dir,name) for name in sorted(os.listdir(img_dir)) if  name.endswith(('.png','.jpg','.webp','.tif','.jpeg'))]
		return imgs_List
	#test()
	else:
		return "Sorry! You got the wrong validation code! Please try again!"

# demo = gr.Interface(sepia, gr.Image(), "image")
demo = gr.Interface(
    Test,
    [
        gr.Radio([f"{a,b,c,d}", "(6,6,5,4)", "(7,4,6,4)"], label="Validation", info=f"Please choose the right validation code before you start!    Prompt  :  {a,b,c,d}"),
    ],
    outputs=gr.Gallery(label="Diffusion"),
	title="PixeR",
    description="Are you worried about lacking of materials for designing a pixel styple game? Here comes a solution! Our project will generate a group of 16 x 16 pages with total number of 32. Feel free to use our tool to generate the materials you need!",
)
demo.launch(share=True)