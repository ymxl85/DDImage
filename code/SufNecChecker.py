import numpy as np
import PIL.Image as Image
import os
import tensorflow as tf
import tensorflow_hub as hub

classifier_url ="https://hub.tensorflow.google.cn/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}
IMAGE_SHAPE = (224, 224)
classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE + (3,))
])

labels_path = '/home/zxd/Desktop/explanation/ImageNetLabels.txt'
imagenet_labels = np.array(open(labels_path).read().splitlines())

data_floder = "/home/zxd/Desktop/explanation/original-image/111/"
input_dd = "/home/zxd/Desktop/explanation/result/output/baylime/"
img_factual = os.listdir(data_floder)
print("img_factual len is ", len(img_factual))
counter_baylime = 0
Sufficiency_baylime = 0

for i in img_factual:
    image = Image.open(data_floder + i)
    image = image.resize(IMAGE_SHAPE)
    image = np.array(image) / 255.0
    image = image[:, :, 0:3]

    result = classifier.predict(image[np.newaxis, ...]) 
    predicted_class = np.argmax(result[0], axis=-1)
    print("prediction:", predicted_class)
    print("predict class is ", imagenet_labels[predicted_class])

    counter = Image.open(input_dd + 'counter' + i)
    counter = counter.resize(IMAGE_SHAPE)
    counter = np.array(counter) / 255.0
    counter = counter[:, :, 0:3]

    suffericient = Image.open(input_dd + 'sufferi' + i)
    suffericient = suffericient.resize(IMAGE_SHAPE)
    suffericient = np.array(suffericient) / 255.0
    suffericient = suffericient[:, :, 0:3]

    if np.argmax(classifier.predict(counter[np.newaxis, ...])) != predicted_class:
        counter_baylime += 1
    if np.argmax(classifier.predict(suffericient[np.newaxis, ...])) == predicted_class:
        Sufficiency_baylime += 1
    print("counter is ", imagenet_labels[np.argmax(classifier.predict(counter[np.newaxis, ...]))])
    print("suffericient is ", imagenet_labels[np.argmax(classifier.predict(suffericient[np.newaxis, ...]))])

print(counter_baylime, Sufficiency_baylime)