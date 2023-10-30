from PIL import Image
from numpy import average, linalg, dot
import os
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

# cosin similarity
def get_thumbnail(image, size=(1200, 750), greyscale=False):
    # image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        image = image.convert('L')
    return image


def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thumbnail(image1)
    image2 = get_thumbnail(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    res = dot(a / a_norm, b / b_norm)
    return res
def save_data(name,sim):
    f = open('/home/zxd/Desktop/explanation/Data/Similarity/sedc_resnet_roaming_shear_similarity.txt', mode='a')  # 
    f.writelines([name, ' ',(str)(sim)])  # 
    f.write('\n')  # write 写入
    f.close()

if __name__ == '__main__':

    input_dir = "/home/zxd/Desktop/explanation/R-Explanation/SEDC/ResNet-Roaming-Shear/"
    image1 = "/home/zxd/Desktop/explanation/R-Explanation/SEDC/ResNet-Roaming-All-Shear/"
    image2 = "/home/zxd/Desktop/explanation/R-Explanation/SEDC/ResNet-Roaming-Shear/"
    a = os.listdir(input_dir)
    count = 0  # 

    for i in a:
        if not i.find("counter"):
            continue
        image3 = image1 +"/"+ i
        if not os.path.exists(image3):
            count += 1
            continue  
        image4 = image2 +"/"+ i
        print(image4)
        print(image3)
        image5 = Image.open(image3)
        image6 = Image.open(image4)
        cosin = image_similarity_vectors_via_numpy(image5, image6)
        save_data(i,cosin)
        print(cosin)
    print("count:",count)
