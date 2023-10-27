import os
from PIL import Image
input_dir = "/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-Roaming/"
out_dir = "/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-Roaming-All-XuanZhuan/"
a = os.listdir(input_dir)
s=0
for i in a:
    s = s+1
    I = Image.open(input_dir+i)
    out = I.transpose(Image.FLIP_LEFT_RIGHT)
    out.save(out_dir+i)
