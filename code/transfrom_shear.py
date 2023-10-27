import os
import cv2
import numpy as np
from wand.image import Image

input_dir = "/home/zxd/Desktop/explanation/R-Explanation/DDImage/ResNet-Roaming/"
out_dir = "/home/zxd/Desktop/explanation/R-Explanation/DDImage/ResNet-Roaming-All-Shear/"
a = os.listdir(input_dir)
s=0
for i in a:
    s = s+1
    image_path = input_dir + i
    out_path = out_dir + i
    with Image(filename = image_path) as image:
        with image.clone() as shear:
            shear.shear('Black', 20)
            shear.resize(224, 224)
            shear.save(filename = out_path)
