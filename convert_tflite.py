import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import tensorflow_model_optimization as tfmot
#load_ext tensorboard

from yolov4.models_v import (
  Mish,
  yolo_anchors_608,
  yolo_anchor_masks,
  yolo_boxes,
  yolo_nms,
)

from tensorflow.keras.layers import (
  Lambda,
  Input,
)

def get_gzipped_model_size(file):
  import zipfile
  import tempfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)
 
size = 608
channels = 3
path = './checkpoints/yolov4.h5'
anchors = yolo_anchors_608
masks = yolo_anchor_masks
label_path = './data/coco.names'
#label_path = './data/voc2012.names'

labels = [c.strip() for c in open(label_path).readlines()]

num_classes = len(labels)

model = tf.keras.models.load_model('./checkpoints/yolov4.h5', compile = False)
#model = tf.keras.models.load_model('./checkpoints/yolov4_2.h5',custom_objects={'Mish':Mish}, compile=False)

#print('output', [a.shape for a in model.predict(img)])

x = inputs = Input([size, size, channels], name='inputs_v')

output_v4_0, output_v4_1, output_v4_2 = model(x)
x = output_v4_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[0]), num_classes + 5)), name ='output_v4_0_137')(output_v4_0)
x = output_v4_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[1]), num_classes + 5)), name = 'output_v4_1_147')(output_v4_1)
x = output_v4_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[2]), num_classes + 5)), name = 'output_v4_2_157')(output_v4_2)

boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes), name = 'yolo_boxes_v4_0')(output_v4_0)
boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes), name = 'yolo_boxes_v4_1')(output_v4_1)
boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], num_classes), name = 'yolo_boxes_v4_2')(output_v4_2)
outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes), name = 'yolo_nms')((boxes_v4_2[:3], boxes_v4_1[:3], boxes_v4_0[:3]))
model_best = tf.keras.models.Model(inputs = [inputs], outputs = outputs_v4, name = 'Yolov4_detect')

converter = tf.lite.TFLiteConverter.from_keras_model(model_best)
###
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True
converter.allow_custom_ops = True
###
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
quantized_model_path_1 = './checkpoints/yolov4_quantied_1.tflite'
with open(quantized_model_path_1, 'wb') as f:
  f.write(tflite_model)
#tf.keras.models.save_model(tflite_model, quantized_model_path_1, include_optimizer=False)


print("Size of gzipped baseline Keras model(Training part): %.2f bytes" % (get_gzipped_model_size('./checkpoints/yolov4.h5')))
print("Size of gzipped converted Tflite model: %.2f bytes" % (get_gzipped_model_size(quantized_model_path_1)))
