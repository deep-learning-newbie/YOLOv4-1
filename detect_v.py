import time
import cv2
import tensorflow as tf
import numpy as np
from absl import logging

#from yolov4.utils_v import draw_outputs, transform_images
from yolov4.models_v import (
  Mish,
  yolo_anchors_608,
  yolo_anchor_masks,
  yolo_boxes,
  yolo_nms,
  transform_images,
  draw_outputs,
)

from tensorflow.keras.layers import (
  Lambda,
  Input,
)

size = 608
channels = 3
path = './checkpoints/yolov4.h5'
anchors = yolo_anchors_608
masks = yolo_anchor_masks
label_path = './data/coco.names'

labels = [c.strip() for c in open(label_path).readlines()]

num_classes = len(labels)

#model = Yolov4_detect(path)
model = tf.keras.models.load_model('./checkpoints/yolov4.h5', custom_objects = {'Mish':Mish},compile = False)
x = inputs = Input([size, size, channels], name='Yolov4')

output_v4_0, output_v4_1, output_v4_2 = model(x)
x = output_v4_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[0]), num_classes + 5)), name ='output_v4_0_137')(output_v4_0)
x = output_v4_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[1]), num_classes + 5)), name = 'output_v4_1_147')(output_v4_1)
x = output_v4_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[2]), num_classes + 5)), name = 'output_v4_2_157')(output_v4_2)

boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes), name = 'yolo_boxes_v4_0')(output_v4_0)
boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes), name = 'yolo_boxes_v4_1')(output_v4_1)
boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], num_classes), name = 'yolo_boxes_v4_2')(output_v4_2)
outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes), name = 'yolo_nms')((boxes_v4_2[:3], boxes_v4_1[:3], boxes_v4_0[:3]))
model_best = tf.keras.models.Model(inputs = [inputs], outputs = outputs_v4, name = 'Yolov4_detect')

t1 = time.time()

image = './data/6dogs'
output = image + '_out'

img_raw = tf.image.decode_image(open(image + '.jpg', 'rb').read(), channels=3)
img = tf.expand_dims(img_raw, 0)
img = transform_images(img, size)

boxes, scores, classes, nums = model_best(img)

t2 = time.time()
logging.info('time: {}'.format(t2 - t1))

img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
img = draw_outputs(img, (boxes, scores, classes, nums), labels)
cv2.imwrite(output + '.jpg', img)
