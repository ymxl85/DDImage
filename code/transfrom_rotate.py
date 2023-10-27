import os
from PIL import Image
import numpy as np

input_dir = "/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-Roaming-All-Rotate/"
out_dir = "/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-Roaming-All-Rotate-back/"

IMAGE_SHAPE = (224, 224)
a = os.listdir(input_dir)
s=0

for i in a:
    s = s+1
    I = Image.open(input_dir+i)
    I = I.resize(IMAGE_SHAPE)
    out = I.rotate(-10, expand=True)  
    out.save(out_dir+i)
