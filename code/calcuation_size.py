import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


def save_data(name,size):
    f = open("/home/zxd/Desktop/explanation/Data/Size/output_dd_size_mobilenet_v2_roaming_segments=slic.txt", mode="a")
    f.writelines([name,' ',(str)(size)])
    f.write('\n')
    f.close()


def cal_size(filename):
    im1 = Image.open(filename)
    x = im1.size[0]
    y = im1.size[1]
    im2 = ImageOps.grayscale(im1)
    prior_img = np.array(im2)

    trigger_mask = np.zeros((y,x),dtype=int)
    union = np.logical_or(prior_img,trigger_mask)
    size = np.count_nonzero(union)/(x*y)
    return size

def cal_size_2(filename):
    im1 = Image.open(filename)
    x = im1.size[0]
    y = im1.size[1]
    mid =0
    all=0
    for i in range(x):
        for j in range(y):
            if im1.getpixel((i,j))!=(127,127,127):
                mid+=1
            all+=1
    return mid/all

def get_baylime_imagenet_data_size(filename):
    for file in os.listdir(filename):
        if file.find("counter") !=-1:
            continue
        size = cal_size_2(os.path.join(filename,file))
        save_data(file,size)

def get_baylime_imagenet_data_size_2(filename):
    nums = 0
    for path, sudbirs, files in os.walk(filename):
        # print(path)
        # print(files)
        for file in files:
            if file.find("suffericient") !=-1:
                print(file)
                size = cal_size(os.path.join(filename, file))
                print(size)
                save_data(file, size)

def get_sedc_imagenet_data_size(filename):
    nums=0
    for path,sudbirs,files in os.walk(filename):
        for file in files:
            if file.find("counter") == -1:
                size = cal_size(os.path.join(filename, file))
                print(size)
                save_data(file,size)

def get_dd_imagenet_data_size(filename):
    nums=0
    for path,sudbirs,files in os.walk(filename):
        for file in files:
            if file.find("suffer") == 0:
                nums +=1
                print(file)
                size = cal_size(os.path.join(filename, file))
                print(size)
                save_data(file,size)

def get_dd_lesser_panada_data_size(filename):
    root_dir = "/home/zxd/Desktop/explanation"
    IMAGE_SHAPE = (224, 224)
    num = 0
    for file in os.listdir(filename):
        # if len(file)>=29:
        # print(file)
        for image in os.listdir(root_dir + "/" + file):
            if image.find("over") != -1:
                # print(image)
                size = cal_size(root_dir + "/" + file + "/" + image)
                print(size)
                save_data(file,size)
def test(image):
    image = Image.open("/home/zxd/Desktop/explanation/original-image/imagenet/ILSVRC2012_val_00000003.JPEG")
    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = image[:, :, 0:3]

    image1 = Image.open("/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-ImageNet/counterILSVRC2012_val_00000003.JPEG")
    image1 = image1.resize((224, 224))
    image1 = np.array(image1)/255.0
    image1 = image1[:, :, 0:3]
    perturbed_image = cv2.GaussianBlur(image, (31, 31), 0)
    perturbed_image = Image.fromarray((perturbed_image * 255).astype(np.uint8))
    image1 = Image.fromarray((image1 * 255).astype(np.uint8))
    plt.imshow(perturbed_image)
    plt.show()
    plt.imshow(image1)
    plt.show()
    print("perturbed_image",perturbed_image)
    print("image1",image1)
    mid = 0
    all = 0
    # print(image1)
    # print(image)
    for i in range(224):
        for j in range(224):
            print(perturbed_image.getpixel((i,j)))
            if image1.getpixel((i,j)) == perturbed_image.getpixel((i,j)):
                mid +=1
            all+=1
    print("mid",mid)
    print("all",all)
    print("mid / all:",mid / all)
    return mid / all
if __name__ == '__main__':
    get_dd_imagenet_data_size("/home/zxd/Desktop/explanation/R-Explanation/DDImage/MobileNet-Roaming-Segments=SLIC/")
    # get_sedc_imagenet_data_size("/home/zxd/Desktop/explanation/R-Explanation/SEDC/InceptionV3-ImageNet-All/")
    # get_baylime_imagenet_data_size_2("/home/zxd/Desktop/explanation/R-Explanation/BayLIME/ResNet-ImageNet-All/")
    
