import time
import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import os
from os import listdir
import pandas as pd
import operator
import tensorflow as tf
import tensorflow_hub as hub
import cv2

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

# classifier_url ="https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
classifier_url = "https://hub.tensorflow.google.cn/google/imagenet/resnet_v2_50/classification/5"
# classifier_url = "https://hub.tensorflow.google.cn/google/imagenet/inception_v3/classification/5"


IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])
labels_path = '/home/zxd/Desktop/explanation/ImageNetLabels.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())

def split(c,n):
    subsets = []
    start =0
    for i in range(n):
        subset = c[start:(start+int((len(c)-start)/(n-i)))]
        subsets.append(subset)
        start = start + len(subset)
    return subsets

def __listmins(c1,c2):
    # return a list of all elements of c1 that are not in c2
    s2 = {}
    for delta in c2:
        s2[delta] =1
    c =[]
    for delta in c1:
        if not s2.get(delta):
            c.append(delta)
    return c

def _listintersect(c1,c2):
    # rteurn the common elements of c1 and c2
    s2 = {}
    for delta in c2:
        s2[delta] = 1
    c = []
    for delta in c1:
        if s2.get(delta):
            c.append(delta)
    return c

def _listunion(c1,c2):
    s2 = {}
    for delta in c1:
        s2[delta] = 1
    c = c1[:]
    for delta in c2:
        if not s2.get(delta):
            c.append(delta)

def dd(image, classifier,segments,output_dd):
    result = classifier.predict(image[np.newaxis, ...])
    predicted_class = np.argmax(result)
    # print("predicted_class is ", predicted_class)
    perturbed_image = cv2.GaussianBlur(image, (31, 31), 0)
    run = 1
    # cbar_offsert = 0
    c = []
    for i in np.unique(segments):
        c.append(i)
    n = 2
    count=1;
    while 1:

        if n > len(c):
            # print("dd done")
            return c
        cs = split(c,n)
        # print("dd (run # "+(str(run))+")-trying split "+(str(n))+": ",cs)

        c_failed =0
        cbar_failed =0
        next_c = c[:]
        next_n = n
        # print("c is ",c)
        for i in range(n):
            # print("pation is ",cs[i])
            t1,t2 = test_classifiers(image,segments,cs[i],classifier,perturbed_image,count,output_dd)
            count += 1
            if t1 == predicted_class and t2 != predicted_class:
                c_failed =1
                next_c = cs[i]
                next_n = 2
                # cbar_offsert =0
                break
        if not c_failed:
            for j in range(n):
                cbars = __listmins(c,cs[j])
                
                t1,t2 = test_classifiers(image,segments,cbars,classifier,perturbed_image,count,output_dd)
                count += 1
                if t1 == predicted_class and t2 != predicted_class:
                    cbar_failed = 1
                    next_c = cbars
                    next_n = n-1
                    # cbar_offsert = i
                    break
        if not c_failed and not cbar_failed:
            if n >= len(c):
                # print("dd done")
                return c
            next_n = min(len(c),n*2)
            # cbar_offsert = (cbar_offsert*next_n)/n

        c = next_c
        run = run + 1
        n = next_n
        # print("\n")

def test_classifiers(image,segments,csub,classifier,perturbed_image,count,output_dd):
    test_image_1 = image.copy()
    test_image_2 = perturbed_image.copy()

    
    for i in csub:
        test_image_1[segments == i] = perturbed_image[segments==i]
        test_image_2[segments == i] = image[segments == i]
    
    result_1 = classifier.predict(test_image_1[np.newaxis,...])
    result_2 = classifier.predict(test_image_2[np.newaxis,...])

    c1 = np.argmax(result_1)
    c2 = np.argmax(result_2)
    # print(imagenet_labels[c1],imagenet_labels[c2])
    return c2,c1

def savetime(name,time):
    f = open("time/output_dd_resnet_v2_50_roaming_shear111.txt", mode="a")
    f.writelines([name,' ',(str)(time)])
    f.write('\n')
    f.close()

if __name__ == '__main__':
    data_floder = "/home/zxd/Desktop/explanation/original-image/Shear_images_roaming/"
    output_dd = "R-Explanation/DDImage/1"
    img_factual = os.listdir(data_floder)
    print("img_factual len is ",len(img_factual))
    for i in img_factual:
        ori_image = "/home/zxd/Desktop/explanation/original-image/roaming_panda_images_vgg16_mobilenet/"+i
        ori = Image.open(ori_image)
        ori = ori.resize(IMAGE_SHAPE)
        ori = np.array(ori) / 255.0
        ori = ori[:, :, 0:3]
        print(i)
        image = Image.open(data_floder+i)
        image = image.resize(IMAGE_SHAPE)
        image = np.array(image)/255.0
        image = image[:,:,0:3]

        # Classify image
        result = classifier.predict(image[np.newaxis, ...])  ##jmy? ...
        predicted_class = np.argmax(result[0], axis=-1)

        if imagenet_labels[predicted_class] != "lesser panda" or \
                imagenet_labels[np.argmax(classifier.predict(ori[np.newaxis, ...])[0], axis=-1)] !="lesser panda":
            continue
        print("prediction:",predicted_class)
        print("predict class is ",imagenet_labels[predicted_class])

        # Segment image
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        
        # #Gen Explanation
        start = time.time()
        explanation = dd(image, classifier,segments,output_dd)
        stop = time.time()
        savetime(i, stop - start)
        #
        test_image_1 = image.copy()
        test_image_2 = np.zeros((224, 224, 3))
        for j in explanation:
            print(j)
            test_image_1[segments == j] = 0
            test_image_2[segments == j] = image[segments == j]
        # plt.imshow(test_image_2)
        # plt.show()

        Image.fromarray((test_image_1*255).astype(np.uint8)).save(f'{output_dd}/{"counter"+i}')
        Image.fromarray((test_image_2*255).astype(np.uint8)).save(f'{output_dd}/{"sufferi"+i}')

