import struct

from absl import flags
from absl.flags import FLAGS
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model
from itertools import repeat
from absl import logging
#import struct

from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

yolo_anchors_416 = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                         (59, 119), (116, 90), (156, 198), (373, 326)],
                        np.float32) / 416
                        
yolo_anchors_608 = np.array([(12, 16), (19, 36), (40, 28), (36, 75), (76, 55),
                         (72, 146), (142, 110), (192, 243), (459, 401)],
                        np.float32) / 608

yolo_anchors_512 = np.array([(12, 16), (19, 36), (40, 28), (36, 75), (76, 55),
                         (72, 146), (142, 110), (192, 243), (459, 401)],
                        np.float32) / 512

#yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_anchor_masks = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

yolo_tiny_anchors = np.array([(10, 14), (23, 27), (37, 58),
                              (81, 82), (135, 169),  (344, 319)],
                             np.float32) / 416
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


yolo_max_boxes = 100
yolo_iou_threshold = 0.5
yolo_score_threshold = 0.5

class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                print("reading 64 bytes")
                w_f.read(8)
            else:
                print("reading 32 bytes")
                w_f.read(4)

            transpose = (major > 1000) or (minor > 1000)
            
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model):
        count = 0
        ncount = 0
        for i in range(161):
            #print('conv_'+str(i))
            try:

                conv_layer = model.get_layer('convn_' + str(i))
                filter = conv_layer.kernel.shape[-1]
                nweights = np.prod(conv_layer.kernel.shape) # kernel*kernel*c*filter
                
                print("loading weights of convolution #" + str(i)+ "- nb parameters: "+str(nweights+filter))             
                
                if i  in [138, 149, 160]:
                    print("Special processing for layer "+ str(i))
                    bias  = self.read_bytes(filter) # bias
                    weights = self.read_bytes(nweights) # weights
                
                else:                    
                    bias  = self.read_bytes(filter) # bias
                    scale = self.read_bytes(filter) # scale
                    mean  = self.read_bytes(filter) # mean
                    var   = self.read_bytes(filter) # variance
                    weights = self.read_bytes(nweights) # weights
                    
                    bias = bias - scale  * mean / (np.sqrt(var + 0.00001)) #normalize bias

                    weights = np.reshape(weights,(filter,int(nweights/filter)))  #normalize weights
                    A = scale / (np.sqrt(var + 0.00001))
                    A= np.expand_dims(A,axis=0)
                    weights = weights* A.T
                    weights = np.reshape(weights,(nweights))
                

                weights = weights.reshape(list(reversed(conv_layer.get_weights()[0].shape)))                 
                weights = weights.transpose([2,3,1,0])
                
                if len(conv_layer.get_weights()) > 1:
                    a=conv_layer.set_weights([weights, bias])
                else:    
                    a=conv_layer.set_weights([weights])
                
                count = count+1
                ncount = ncount+nweights+filter
             
            except ValueError:
                print("no convolution #" + str(i)) 
        
        print(count, "Conv normalized layers loaded ", ncount, " parameters")
    
    def reset(self):
        self.offset = 0

'''
def read_labels(labels_path):
    with open(labels_path) as f:
        labels = f.readlines()
    labels = [c.strip() for c in labels]
    return labels

def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3
     
        

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
 
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
 
        return self.label
 
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
 
        return self.score
'''


def broadcast_diou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    iou = broadcast_iou(box_1, box_2)

    #broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new shape: (..., N,(x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    # diagonal distance of the smallest enclosing box covering the two boxes
    c_w = tf.maximum(tf.maximum(box_1[..., 2], box_2[..., 2]) -
                       tf.minimum(box_1[..., 0], box_2[..., 0]), 0)
    c_h = tf.maximum(tf.maximum(box_1[..., 3], box_2[..., 3]) -
                       tf.minimum(box_1[..., 1], box_2[..., 1]), 0)
    c = c_w * c_w + c_h * c_w + 1e-8

    # distance between center points in two boxes
    d_w = ((box_1[...,2] + box_1[...,0]) / 2) - ((box_2[...,2] + box_2[...,0]) / 2)
    d_h = ((box_1[...,1] + box_1[...,3]) / 2) - ((box_2[...,1] + box_2[...,3]) / 2)
    d = d_w * d_w + d_h * d_h
    diou_term = d / c
    return iou - diou_term

def broadcast_ciou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2)) pred_box
    # box_2: (N, (x1, y1, x2, y2)) ground_truth_box

    iou = broadcast_iou(box_1, box_2)

    #broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new shape: (..., N,(x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)
    box_1_w = tf.maximum(box_1[...,2], box_1[..., 0]) - tf.minimum(box_1[...,2], box_1[..., 0])
    box_1_h = tf.maximum(box_1[...,3], box_1[..., 1]) - tf.minimum(box_1[...,3], box_1[..., 1])

    box_2_w = tf.maximum(box_2[...,2], box_2[..., 0]) - tf.minimum(box_2[...,2], box_2[..., 0])
    box_2_h = tf.maximum(box_2[...,3], box_2[..., 1]) - tf.minimum(box_2[...,3], box_2[..., 1])

    # diagonal distance for smallest enclosing box cover the two boxes
    c_w = tf.maximum(tf.maximum(box_1[..., 2], box_2[..., 2]) -
                       tf.minimum(box_1[..., 0], box_2[..., 0]), 0)
    c_h = tf.maximum(tf.maximum(box_1[..., 3], box_2[..., 3]) -
                       tf.minimum(box_1[..., 1], box_2[..., 1]), 0)
    c = c_w * c_w + c_h * c_w

    if c == 0:
        return iou

def broadcast_iou(box_1, box_2):
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
        (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
        (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def draw_outputs(img, outputs, class_names):
    boxes, objectness, classes, nums = outputs
    boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
    wh = np.flip(img.shape[0:2])
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, '{} {:.4f}'.format(
            class_names[int(classes[i])], objectness[i]),
            x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return img


def draw_labels(x, y, class_names):
    img = x.numpy()
    boxes, classes = tf.split(y, (4, 1), axis=-1)
    classes = classes[..., 0]
    wh = np.flip(img.shape[0:2])
    for i in range(len(boxes)):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
        img = cv2.putText(img, class_names[classes[i]],
                          x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          1, (0, 0, 255), 2)
    return img


def freeze_all(model, frozen=True):
    model.trainable = not frozen
    if isinstance(model, tf.keras.Model):
        for l in model.layers:
            freeze_all(l, frozen)

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    # 'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size):
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, (size, size))

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    paddings = [[0, yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=416):
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/family.jpg', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

basic_activation = "Mish"

class Mish(tf.keras.layers.Layer):
    """
    ..math::
        Mish(x) = x*tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    
    Example:
        X_input = Input(input_shape)
        X = Mish()(X_input)
    """
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True
    
    def call(self, inputs):
        #return inputs * tf.math.tanh(tf.math.softplus(inputs))
        return inputs * tf.math.tanh(tf.math.log(1. + tf.math.exp(inputs))) # for converting tflite format
    
    def get_config(self):
        base_config = super(Mish, self).get_config()
        return {**base_config}

    def compute_output_shape(self, input_shape):
        return input_shape

def _conv_block(inp, convs, skip=False):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        
        if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)), name='zerop_' + str(conv['layer_idx']))(x)  # peculiar padding as darknet prefer left and top
        
        x = Conv2D(conv['filter'], 
                   conv['kernel'], 
                   strides=conv['stride'], 
                   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                   name='convn_'+ str(conv['layer_idx']) if conv['bnorm'] else 'conv_' + str(conv['layer_idx']),
                   use_bias=True)(x)
        
        if conv['bnorm']: x = BatchNormalization(name='BN_' + str(conv['layer_idx']))(x)    
        
        if conv['activ'] == 1: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
        if conv['activ'] == 2: x = Mish()(x)#Activation('mish', name='mish_' + str(conv['layer_idx']))(x) 
            
    return Add(name='add_' + str(conv['layer_idx']+1))([skip_connection, x]) if skip else x

def make_yolov4_model(size, classes_nb):
        
    input_image = Input(shape=(size, size, 3), name='input_0')

    # ---------------------------------- begin BACKBONE CSP-Darnet 53 -----------------------------------------------------------
    # Layer  0
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 0}])

    # Layer  1
    x = _conv_block(x, [{'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 2, 'layer_idx': 1}])
    layer_1 = x
   
    # ---------- begin 1*Conv+conv+residual 304*304    
    # Layer  2 
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 2}])
    layer_2 = x
    

    # route  1 (layers = -2)
    x = layer_1
    # Layer  3 => 5
    x = _conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 4},                  
                        {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 5},
                        {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 6}],
                   skip = True)

    # Layer  8 => 8
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 8}])
    layer_8 = x
    
    # route  8+2 (layers = -1, -7)
    x = Concatenate(name='concat_9')([layer_8, layer_2])
    
    # Layer 10 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 10},
                        {'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 2, 'layer_idx': 11}])
    layer_11 = x
    # ---------- end 1*Conv+conv+residual 304*304
   
    # ---------- begin 2*Conv+conv+residual 152*152      
    # Layer  12
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 12}])
    layer_12 = x
    
  
    # route  11 (layers = -2)
    x = layer_11
    # Layer 14 => 16
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 14},                    
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 15},
                        {'filter':  64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 16}],
                   skip = True)
    
    # Layer 18 => 19
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 18},
                        {'filter':  64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 19}],
                   skip = True)
    
    # Layer  21
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 21}]) 
    layer_21 = x
    
    # route  21+12 (layers = -1,-10)
    x = Concatenate(name='concat_22')([layer_21, layer_12])
    
    # Layer 23 => 24
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 23},
                        {'filter':  256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 2, 'layer_idx': 24}])
    layer_24 = x
    # ---------- end 2*Conv+conv+residual 152*152        
    
    # ---------- begin 8*Conv+conv+residual 76*76    
    # Layer  25
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 25}])
    layer_25 = x
        
    
    # route  24 (layers = -2)
    x = layer_24
    
    # Layer 27 => 29
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 27},                       
                        {'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 28},
                        {'filter':  128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 29}],
                   skip = True)
    
    
    # Layer 31 => 50
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 31+(i*3)},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 32+(i*3)}],
                       skip = True)
  
    # Layer  52
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 52}])
    layer_52 = x
        
    # route  52+25 (layers = -1,-28)
    x = Concatenate(name='concat_53')([layer_52, layer_25])
    
 
    # Layer 54
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 54}])
    # ---------- initial output 76*76 
    layer_54 = x
    
    # Layer  55
    x = _conv_block(x, [{'filter':  512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 2, 'layer_idx': 55}])
    layer_55 = x
    # ---------- end 8*Conv+conv+residual 76*76        
    
    # ---------- begin 8*Conv+conv+residual 38*38  
    # Layer  56
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 56}])
    layer_56 = x
    
    # route  55 (layers = -2)
    x = layer_55
    
    # Layer 58 => 60
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 58},
                        {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 59},
                        {'filter':  256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 60}],
                   skip = True)     
    
    # Layer 62 => 81
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 62+(i*3)},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 63+(i*3)}],
                       skip = True)

    # Layer  83
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 83}])
    layer_83 = x

    # route  83+56 (layers = -1,-28)
    x = Concatenate(name='concat_84')([layer_83, layer_56])
    
    # Layer 85
    x = _conv_block(x, [{'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 85}])
    # ---------- initial output 38*38 
    layer_85 = x
    
    # Layer  86
    x = _conv_block(x, [{'filter':  1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 2, 'layer_idx': 86}])
    layer_86 = x
    
    # Layer  87
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 87}])
    layer_87 = x
        
    # route  86 (layers = -2)
    x = layer_86
    # ---------- end 8*Conv+conv+residual 38*38  
    
    # ---------- begin 4*Conv+conv+residual 19*19 
    # Layer 89 => 92
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 89},                         
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 90},
                        {'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 91}],
                   skip = True) 
    
    
    # Layer 93 => 100
    for i in range(3):
        x = _conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 93+(i*3)},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 94+(i*3)}],
                       skip = True)  
    
    # ---------- end 4*Conv+conv+residual 19*19 
    
    # Layer  102 => 102
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 102}])  
    layer_102 = x
    
    # route  102+87 (layers = -1,-16)
    x = Concatenate(name='concat_103')([layer_102, layer_87])
    
    # Layer 104 => 107
    x = _conv_block(x, [{'filter':  1024, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 2, 'layer_idx': 104},
                        {'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 105},
                        {'filter':  1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 106},                        
                        {'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 107}])
    layer_107 = x 
    # ---------------------------------- end BACKBONE CSP-Darnet 53 -----------------------------------------------------------
    
    # ---------------------------------- begin SPP part (Spatial Pyramid Pooling layer) ---------------------------------------
    # Layer 108
    x =MaxPool2D(pool_size=(5, 5), strides=1, padding='same', name = 'layer_108')(x)  
    layer_108 = x
    
    # route  107 (layers = -2)
    x = layer_107
    
    # Layer 110
    x =MaxPool2D(pool_size=(9, 9), strides=1, padding='same', name = 'layer_110')(x)    
    layer_110 = x
    
    # route  107 (layers = -4)
    x = layer_107
        
    # Layer 112
    x =MaxPool2D(pool_size=(13, 13), strides=1, padding='same', name = 'layer_112')(x) 
    layer_112 = x
    
    # route  112+110+108+107 (layers=-1,-3,-5,-6)
    x = Concatenate(name='concat_113')([layer_112, layer_110, layer_108, layer_107])
    # ---------------------------------- end SPP part (Spatial Pyramid Pooling layer) ---------------------------------------
 

    
    # Layer 114 => 116
    x = _conv_block(x, [{'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 114},
                        {'filter':  1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 115},
                        {'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 116}])
    layer_116 = x
                        
    # Layer 117                    
    x = _conv_block(x, [{'filter':   256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 117}])
    layer_117 = x
    # Layer 118
    x = UpSampling2D(size=(2, 2), name = 'upsamp_118')(x)
    layer_118 = x
                        
    # route  85 (layers = 85)
    x = layer_85
    
    # Layer 120
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 120}])
    layer_120 = x
                        
    # route  120+118 (layers = -1, -3)
    x = Concatenate(name='concat_121')([layer_120, layer_118])
    layer_121 = x    
    
  
    # Layer 122 => 126
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 122},
                        {'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 123},
                        {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 124},  
                        {'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 125},
                        {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 126}])
    layer_126 = x 
    
    
    # Layer 127                    
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 127}])
    layer_127 = x
    # Layer 128
    x = UpSampling2D(size=(2, 2), name = 'upsamp_128')(x)
    layer_128 = x
    
    # ---------------------------------- begin head output 76*76 ---------------------------------------                   
    # route  54 (layers = 54)
    x = layer_54
    
    # Layer 130
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ': 1, 'layer_idx': 130}])
    layer_130 = x
                        
    # route  130+128 (layers = -1, -3)                 
    x = Concatenate(name='concat_131')([layer_130, layer_128])
    layer_131 = x
    
    # -- begin Convulationnal set 76*76
    # Layer 132 => 136
    x = _conv_block(x, [{'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 132},
                        {'filter':  256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 133},
                        {'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 134},  
                        {'filter':  256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 135},
                        {'filter':  128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 136}])
    layer_136 = x                   
    # -- end Convulationnal set 76*76
    
    # -- beging last Convulationnal 3*3 and 1*1 for 76*76
    # Layer 137 => 138
    x = _conv_block(x, [{'filter':  256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 137}]) 
    layer_137 = x 
    x = _conv_block(x, [{'filter':  3*(classes_nb + 5), 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':  0, 'layer_idx': 138}])   
    # -- end last Convulationnal 3*3 and 1*1 for 76*76
    
    # -- output 76*76 
    # Layer 139
    yolo_139 = x
    # ---------------------------------- end head output 76*76 ---------------------------------------
    
    # ---------------------------------- begin head output 38*38 ---------------------------------------
    # route  136 (layers = -4)
    x = layer_136
    
    # Layer 141
    x = _conv_block(x, [{'filter':  256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 1, 'layer_idx': 141}])
    layer_141 = x
                        
    # route  141+126 (layers = -1, -16)                   
    x = Concatenate(name='concat_142')([layer_141, layer_126])
    
    # -- begin Convulationnal set 38*38
    # Layer 143 => 147
    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':   1, 'layer_idx': 143},
                        {'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':   1, 'layer_idx': 144},
                        {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':   1, 'layer_idx': 145},  
                        {'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':   1, 'layer_idx': 146},
                        {'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':   1, 'layer_idx': 147}])  
    layer_147 = x
    
    # -- end Convulationnal set 38*38
    
    # -- beging last Convulationnal 3*3 and 1*1 for 38*38
    # Layer 148 => 149                    
    x = _conv_block(x, [{'filter':  512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':  1, 'layer_idx': 148},
                        {'filter':  3*(classes_nb + 5), 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':  0, 'layer_idx': 149}])
    # -- end last Convulationnal 3*3 and 1*1 for 38*38
    
    # -- output 38*38  
    # Layer 150
    yolo_150 = x   
    # ---------------------------------- end head output 38*38 ---------------------------------------
    
    # ---------------------------------- begin head output 19*19 ---------------------------------------
    # route  147 (layers = -4)
    x = layer_147
        
    # Layer 152
    x = _conv_block(x, [{'filter':  512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'activ': 1, 'layer_idx': 152}])
    layer_152 = x  
                        
    # route  152+166 (layers = -1, -37)                   
    x = Concatenate(name='concat_153')([layer_152, layer_116]) 
                        
    # -- begin Convulationnal set for 19*19                    
    # Layer 154 => 160
    x = _conv_block(x, [{'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 154},
                        {'filter':  1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 155},
                        {'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 156},
                        {'filter':  1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 157},  
                        {'filter':   512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 158}, 
    # -- end Convulationnal set for 19*19
                        
    # -- beging last Convulationnal 3*3 and 1*1 for 19*19                   
                        {'filter':  1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'activ':    1, 'layer_idx': 159},
                        {'filter':  3*(classes_nb + 5), 'kernel': 1, 'stride': 1, 'bnorm': True, 'activ':    0, 'layer_idx': 160}])  
    # -- end last Convulationnal 3*3 and 1*1 for 19*19                     
    
    # -- output 19*19                    
    # Layer 161
    yolo_161 = x
    # ---------------------------------- end head output 19*19 ---------------------------------------                    
                        
    model = Model(input_image, [yolo_139, yolo_150, yolo_161], name = 'Yolo_v4')    
    return model

#def Mish(inputs):
#  return inputs * tf.math.tanh(tf.math.softplus(inputs))

tf.keras.utils.get_custom_objects().update({'Mish':Mish})
'''
def DarknetConv(x, filters, size, strides=1, activation=basic_activation, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias= not batch_norm, kernel_regularizer=l2(0.0005),
               kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
               bias_initializer = tf.constant_initializer(0.))(x)
    
    if batch_norm:
        x = BatchNormalization()(x)

    if activation == "Mish":
        x = Mish()(x)
    elif activation == "LeakyReLU":
        x = LeakyReLU(alpha=0.1)(x)
    
    elif activation == "Linear":
        x = x
    else:
        print("Check the DarknetConv's batch_norm")
        exit()
    return x

#def YoloV4(name = )

def DarknetResidual(x, filters, activation = basic_activation, batch_norm = True):
    prev = x
    x = DarknetConv(x, filters // 2, 1, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, filters, 3, activation = activation, batch_norm = batch_norm)
    x = Add()([prev, x])
    #x = prev + x
    return x

def csdarknet_residual(input, filters, activation = basic_activation ,batch_norm = True):
    previous = input
    x = DarknetConv(input, filters, 1, activation = activation, strides = 1)
    x = DarknetConv(x, filters, 3, 1, activation = activation)
    x = Add()([previous, x])
    #x = previous + x
    return x

def DarknetBlock(x, filters, blocks, activation = basic_activation, batch_norm = True):
    x = DarknetConv(x, filters, 3, strides=2, activation = activation, batch_norm = batch_norm)
    for _ in range(blocks):
        x = DarknetResidual(x, filters, activation = activation, batch_norm = batch_norm)
    return x

def csdarknet_residual_block(input, filters, blocks, activation = basic_activation, batch_norm = True):
    x = DarknetConv(input, filters, 1, activation = activation, batch_norm = batch_norm)
    for _ in repeat(None, blocks):
        x = csdarknet_residual(x, filters, activation = activation, batch_norm = batch_norm)
    return x

def csdarknet_block(input, filters, blocks, activation = basic_activation, batch_norm = True):
    x = previous_1 = DarknetConv(input, 2*filters, 3, strides = 2, activation = activation, batch_norm = batch_norm)
    x = previous_2 = DarknetConv(x, filters, 1, strides = 1, activation = activation, batch_norm = batch_norm)

    x = csdarknet_residual_block(previous_1, filters, blocks, activation = activation, batch_norm = batch_norm)
    
    x = DarknetConv(x, filters, 1, strides = 1, activation = activation, batch_norm = batch_norm)
    x = Concatenate()([previous_2, x])
    y = x
    x = DarknetConv(x, filters * 2, 1, strides = 1, activation = activation, batch_norm = batch_norm)
    return x, y

def spp(input):
    x = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same')(input)
    previous_1 = x

    x = MaxPool2D(pool_size = (9,9), strides = 1, padding = 'same')(input)
    previous_2 = x
    x = MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same')(input)

    previous_3 = x
    x = Concatenate()([previous_3, previous_2, previous_1, input])

    return x

def Conv(x, filters, size, strides=1, activation=basic_activation, batch_norm=True, idx = None):#, layer_idx = None):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)), name = 'ZeroPadding2D_' + str(idx))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005),
               kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
               bias_initializer = tf.constant_initializer(0.),
               name = 'Conv_'+ str(idx))(x)
    
    if batch_norm:
        x = BatchNormalization(name = 'BN_' + str(idx))(x)

    if activation == "Mish":
        x = Mish(name = 'mish_' + str(idx))(x)
    elif activation == "LeakyReLU":
        x = LeakyReLU(alpha=0.1, name = 'leakyrelu_' + str(idx))(x)
    
    elif activation == "Linear":
        x = x
    else:
        print("Check the DarknetConv's batch_norm")
        exit()
    return x

def Yolov4_2(size=None, channels=3, anchors=yolo_anchors, masks=yolov4_anchor_masks, classes=80, training=False, activation = basic_activation, batch_norm = True):
    x = inputs = Input([size, size, channels], name = 'input_0')
    x = Conv(x, 32, 3, activation = activation, batch_norm = batch_norm, idx = 1)
    x = Conv(x, 64, 3, 2, activation = activation, batch_norm = batch_norm, idx =2)
    previous_1 = x
    x = previous_2 = Conv(x, 64, 1, activation= activation, batch_norm=batch_norm, idx =3)
    previous_2 = x
    x = previous_1
    x = Conv(x, 64, 1, activation = activation, batch_norm = batch_norm, idx =4)
    previous_3 = x
    x = Conv(x, 32, 1, activation = activation, batch_norm = batch_norm, idx =5)
    x = Conv(x, 64, 3, strides = 1, activation = activation, batch_norm = batch_norm, idx =6) ###

    x = Add(name = 'Add_7')([previous_3, x])
    x = Conv(x, 64, 1, activation = activation, batch_norm = batch_norm, idx =8)
    x = Concatenate(name = 'Concatenate_9')([x, previous_2])

    x = Conv(x, 64, 1, activation = activation, batch_norm = batch_norm, idx =10)

    ###DownSampling

    x = Conv(x, 128, 3, 2, activation = activation, batch_norm = batch_norm, idx =11)
    previous_3 = x

    x = Conv(x, 64, 1, 1, activation = activation, batch_norm = batch_norm, idx =12)
    previous_7 = x
    x = previous_3
    x = Conv(x, 64, 1, 1, activation = activation, batch_norm = batch_norm, idx =13)
    previous_4 = x
    x = Conv(x, 64, 1, 1, activation = activation, batch_norm = batch_norm, idx =14)
    x = Conv(x, 64, 3, 1, activation = activation, batch_norm = batch_norm, idx =15)
    x = Add(name = 'Add_16')([previous_4, x])
    previous_5 = x
    x = Conv(x, 64, 1, 1, activation = activation, batch_norm = batch_norm, idx =17)
    x = Conv(x, 64, 3, 1, activation = activation, batch_norm = batch_norm, idx =18)
    x = Add(name = 'Add_19')([previous_5, x])
    previous_6 = x
    x = Conv(x, 64, 1, 1, activation = activation, batch_norm = batch_norm, idx =20)
    x = Concatenate(name = 'Concatenate_21')([x, previous_7])
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =22)

    ###DownSampling

    x =Conv(x, 256, 3, 2, activation = activation, batch_norm = batch_norm, idx =23)
    previous_8 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =24)
    previous_18 = x

    x = previous_8
    x =Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =25)
    previous_9 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =26)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =27)
    x = Add(name = 'Add_28')([previous_9, x])
    previous_10 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =29)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =30)
    x = Add(name = 'Add_31')([previous_10, x])
    previous_11 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =32)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =33)
    x = Add(name = 'Add_34')([previous_11, x])
    previous_12 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =35)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =36)
    x = Add(name = 'Add_37')([previous_12, x])
    previous_13 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =38)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =39)
    x = Add(name = 'Add_40')([previous_13, x])
    previous_14 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =41)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =42)
    x = Add(name = 'Add_43')([previous_14, x])
    previous_15 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =44)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =45)
    x = Add(name = 'Add_46')([previous_15, x])
    previous_16 = x
    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =47)
    x = Conv(x, 128, 3, 1, activation = activation, batch_norm = batch_norm, idx =48)
    x = Add(name = 'Add_49')([previous_16, x])
    previous_17 = x

    x = Conv(x, 128, 1, 1, activation = activation, batch_norm = batch_norm, idx =50)
    x = Concatenate(name = 'Concatenate_51')([x, previous_18])
    x_54 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =52)

    ###Downsampling

    x = Conv(x, 512, 3, 2, activation = activation, batch_norm = batch_norm, idx =59)
    previous_19 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =60)
    previous_29 = x
    x = previous_19
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =61)
    previous_20 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =62)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =63)
    x = Add(name = 'Add_64')([previous_20, x])
    previous_21 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =65)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =66)
    x = Add(name = 'Add_67')([previous_21, x])
    previous_22 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =68)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =69)
    x = Add(name = 'Add_70')([previous_22, x])
    previous_23 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =71)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =72)
    x = Add(name = 'Add_73')([previous_23, x])
    previous_24 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =74)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =75)
    x  = Add(name = 'Add_76')([previous_24, x])
    previous_25 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =77)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =78)
    x = Add(name = 'Add_79')([previous_25, x])
    previous_26 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =80)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =81)
    x  = Add(name = 'Add_82')([previous_26, x])
    previous_27 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =83)
    x = Conv(x, 256, 3, 1, activation = activation, batch_norm = batch_norm, idx =84)
    x = Add(name = 'Add_85')([previous_27, x])
    previous_28 = x
    x = Conv(x, 256, 1, 1, activation = activation, batch_norm = batch_norm, idx =86)
    x  = Concatenate(name = 'Concatenate_87')([x, previous_29])
    x_85 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =88)

    ###Downsampling

    x = Conv(x, 1024, 3, 2, activation = activation, batch_norm = batch_norm, idx =89)
    previous_30 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =90)
    previous_36 = x
    x = previous_30
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =91)
    previous_31 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =92)
    x = Conv(x, 512, 3, 1, activation = activation, batch_norm = batch_norm, idx =93)
    x = Add(name = 'Add_94')([previous_31, x])
    previous_32 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =95)
    x = Conv(x, 512, 3, 1, activation = activation, batch_norm = batch_norm, idx =96)
    x = Add(name = 'Add_97')([previous_32, x])
    previous_33 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =98)
    x = Conv(x, 512, 3, 1, activation = activation, batch_norm = batch_norm, idx =99)
    x = Add(name = 'Add_100')([previous_33, x])
    previous_34 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =101)
    x = Conv(x, 512, 3, 1, activation = activation, batch_norm = batch_norm, idx =102)
    x = Add(name = 'Add_103')([previous_34, x])
    previous_35 = x
    x = Conv(x, 512, 1, 1, activation = activation, batch_norm = batch_norm, idx =104)
    x = Concatenate(name = 'Concatenate_105')([x, previous_36])
    x = Conv(x, 1024, 1, 1, activation = activation, batch_norm = batch_norm, idx =106)

    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =107)
    x = Conv(x, 1024, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =108)
    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =109)
    previous_37 = x

    ###SPP

    x = previous_37
    
    x = MaxPool2D(pool_size = (5,5), strides = 1, padding = 'same', name = 'maxpool2d_110')(x)
    previous_38 = x
    x = previous_37
    x = MaxPool2D(pool_size = (9,9), strides = 1, padding = 'same', name = 'maxpool2d_111')(x)
    previous_39 = x
    x = previous_37
    x = MaxPool2D(pool_size = (13, 13), strides = 1, padding = 'same', name = 'maxpool2d_112')(x)
    previous_40 = x
    x = Concatenate(name = 'Concatenate_113')([previous_40, previous_39, previous_38, previous_37])

    ###End SPP

    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =114)
    x = Conv(x, 1024, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =115)
    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =116)
    output_1 = x
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =117)

    x = UpSampling2D(2, name = 'UpSampling2D_118')(x)
    previous_41 = x
    x = x_85
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =119)
    x = Concatenate(name = 'Concatenate_120')([x, previous_41])
    #x = Concatenate(name = 'Concatenate_120')([x, previous_41])

    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =121)
    x = Conv(x, 512, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =122)
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =123)
    x = Conv(x, 512, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =124)
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =125)
    output_2 = x
    x = Conv(x, 128, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =126)
    x = UpSampling2D(2, name = 'UpSampling2D_127')(x)
    previous_42 = x
    x = x_54
    x = Conv(x, 128, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =128)
    x = Concatenate(name = 'Concatenate_129')([x, previous_42])
    #previous_44 = x
    x = Conv(x, 128, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =130)
    x = Conv(x, 256, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =131)

    #x = yoloConv_v4(128, name = 'yolo_conv_v4_0')(x)

    x = Conv(x, 128, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =132)
    x = Conv(x, 256, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =133)
    x = Conv(x, 128, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =134)
    previous_43 = x

    x = Conv(x, 256, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =135)
    x = Conv(x, 3*(classes + 5), 1, 1, activation = 'Linear', batch_norm = False, idx =136)
    output_v4_0 = x
    #x = output_v4_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[0]), classes + 5)), name ='output_v4_0_137')(x)

    x = previous_43
    x = Conv(x, 256, 3, 2, activation = 'LeakyReLU', batch_norm = batch_norm, idx =138)
    x = Concatenate(name = 'Concatenate_139')([x, output_2])
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =140)
    x = Conv(x, 512, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =141)
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =142)
    x = Conv(x, 512, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =143)
    x = Conv(x, 256, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =144)
    previous_44 = x
    x = Conv(x, 512, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =145)
    x = Conv(x, 3*(classes + 5), 1, 1, activation = 'Linear', batch_norm = False, idx =146)
    output_v4_1 = x
    #x = output_v4_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[1]), classes + 5)), name = 'output_v4_1_147')(x)
    x = previous_44
    x = Conv(x, 512, 3, 2, activation = 'LeakyReLU', batch_norm = batch_norm, idx = 148)
    x = Concatenate(name = 'Concatenate_149')([x, output_1])
    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =150)
    x = Conv(x, 1024, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =151)
    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =152)
    x = Conv(x, 1024, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =153)
    x = Conv(x, 512, 1, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =154)
    x = Conv(x, 1024, 3, 1, activation = 'LeakyReLU', batch_norm = batch_norm, idx =155)
    x = Conv(x, 3*(classes + 5), 1, 1, activation = 'Linear', batch_norm = False, idx =156)
    output_v4_2 = x
    #x = output_v4_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[2]), classes + 5)), name = 'output_v4_2_157')(x)

    return Model(inputs, [output_v4_0, output_v4_1, output_v4_2], name = 'yolov4')
    #output_v4_0 = Yolov4Output(128, len(masks[2]), classes, name = 'yolo_output_v4_0')(x)
    #x = yoloConv_v4(256, name = 'yolo_conv_v4_1')((x, output_2))
    #output_v4_1 = Yolov4Output(256, len(masks[1]), classes, name = 'yolo_output_v4_1')(x)
    #x = yoloConv_v4(512, name = 'yolo_conv_v4_2')((x, output_1))
    #output_v4_2 = Yolov4Output(512, len(masks[0]), classes, name = 'yolo_output_v4_2')(x)

    #x = output_v4_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[0]), classes + 5)), name ='output_v4_0_137')(x)
    #x = output_v4_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[1]), classes + 5)), name = 'output_v4_1_147')(x)
    #x = output_v4_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[2]), classes + 5)), name = 'output_v4_2_157')(x)
    #if training:
    #    return Model(inputs, [output_v4_2, output_v4_1, output_v4_0], name = 'yolov4')

    #boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name = 'yolo_boxes_v4_0')(output_v4_0)
    #boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name = 'yolo_boxes_v4_1')(output_v4_1)
    #boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name = 'yolo_boxes_v4_2')(output_v4_2)
    #outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name = 'yolo_nms')((boxes_v4_0[:3], boxes_v4_1[:3], boxes_v4_2[:3]))

    #return Model(inputs, outputs_v4, name = 'yolov4')



def csdarknet53_spp(name = None, activation = basic_activation, batch_norm = True):
    x = inputs = Input([None, None, 3])

    x = DarknetConv(x, 32, 3, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, 64, 3, 2, activation = activation, batch_norm = batch_norm)
    previous_1 = x
    x = DarknetConv(x, 64, 1, activation= activation, batch_norm=batch_norm)
    previous_2 = x
    x = DarknetConv(previous_1, 64, 1, activation = activation, batch_norm = batch_norm)
    previous_3 = x
    
    x = DarknetConv(x, 32, 1, activation = activation, batch_norm = batch_norm)
    x = DarknetConv(x, 64, 3, strides = 1, activation = activation, batch_norm = batch_norm) ###

    x = Add(name = 'Add_error')([previous_3, x])
    x = BatchNormalization()(x)
    #x = previous_3 + x
    x = Conv2D(64, kernel_size = 1, strides = 1, padding='same', kernel_regularizer=l2(0.0005), kernel_initializer = tf.random_normal_initializer(stddev = 0.01), bias_initializer = tf.constant_initializer(0.), name = 'Conv2D_add')(x)
    x = BatchNormalization()(x)
    x = Mish()(x)

    #x = DarknetConv(x, 64, 1, activation = activation, batch_norm = False)#########batch_norm = batch_norm
    x = Concatenate()([x, previous_2])

    x = DarknetConv(x, 64, 1, activation = activation, batch_norm = batch_norm)


    x, _ = csdarknet_block(x, 64, 2, activation = activation, batch_norm = batch_norm)
    x, x_54 = csdarknet_block(x, 128, 8, activation = activation, batch_norm = batch_norm)
    x, x_85 = csdarknet_block(x, 256, 8, activation = activation, batch_norm = batch_norm)
    x, _ = csdarknet_block(x, 512, 4, activation = activation, batch_norm = batch_norm)

    x = DarknetConv(x, 512, 1, activation= 'LeakyReLU' )
    x = DarknetConv(x, 1024, 3, activation= 'LeakyReLU')
    x = DarknetConv(x, 512, 1, activation= 'LeakyReLU')

    x = spp(x)

    return Model(inputs = inputs, outputs = (x, x_54, x_85), name = name)


def panet_block(filters):
    def panet_bloc(input, subinput):
        x = DarknetConv(input, filters, 1, activation= "LeakyReLU")
        x = DarknetConv(x, filters * 2, 3, activation= "LeakyReLU")
        x = DarknetConv(x, filters, 1, activation= "LeakyReLU")

        output_1 = x
        x = DarknetConv(x, filters // 2, 1, activation= "LeakyReLU")
        x = previous =UpSampling2D(2)(x)

        x = DarknetConv(subinput, filters // 2, 1, activation= "LeakyReLU")

        x = Concatenate()([x, previous])

        x = DarknetConv(x, filters // 2, 1, activation= "LeakyReLU")
        x = DarknetConv(x, filters, 3, activation= "LeakyReLU")
        return x, output_1
    return panet_bloc

def panet(name = None):
    def panet_model(x_in):
        inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:]), Input(x_in[2].shape[1:])
        main, x_54, x_85 = inputs
        x, output_1 = panet_block(512)(main, x_85)
        x, output_2 = panet_block(256)(x, x_54)

        return Model(inputs, outputs = (x, output_1, output_2), name = name)(x_in)
    return panet_model


def yoloConv_v4(filters, name = None):
    def yolo_conv_v4(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_sub = inputs

            x = Conv(x, filters, 3, strides = 2, batch_norm= "LeakyReLU")
            x = Concatenate()([x, x_sub])
            x = Conv(x, filters, 1, batch_norm= "LeakyReLU")
            x = Conv(x, filters * 2, 3, batch_norm= "LeakyReLU")
        else:
            x = inputs = Input(x_in.shape[1:])

        x = Conv(x, filters, 1, batch_norm= "LeakyReLU")
        x = Conv(x, filters * 2, 3, batch_norm= "LeakyReLU")
        x = Conv(x, filters, 1, batch_norm= "LeakyReLU")
        #nex = x 
        return Model(inputs, x, name = name)(x_in)

    return yolo_conv_v4

def YoloV4(size = None, channels = 3, anchors = yolo_anchors, masks = yolo_anchor_masks, classes = 80, training = False):
    x = inputs = Input([size, size, channels])
    #x, x_54, x_85 = csdarknet53_spp(name = 'csdarknet53_spp')(x)
    x, x_54, x_85 = csdarknet53_spp(name = 'csdarknet53_spp')(x)
    x, output_1, output_2 = panet(name = 'panet')((x, x_54, x_85))
    x = yoloConv_v4(128, name = 'yolo_conv_v4_0')(x)
    output_v4_0 = Yolov4Output(128, len(masks[0]), classes, name = 'yolo_output_v4_0')(x)
    x = yoloConv_v4(256, name = 'yolo_conv_v4_1')((x, output_2))
    output_v4_1 = Yolov4Output(256, len(masks[1]), classes, name = 'yolo_output_v4_1')(x)
    x = yoloConv_v4(512, name = 'yolo_conv_v4_2')((x, output_1))
    output_v4_2 = Yolov4Output(512, len(masks[2]), classes, name = 'yolo_output_v4_2')(x)
    if training:
        return Model(inputs, [output_v4_2, output_v4_1, output_v4_0], name = 'yolov4')

    boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name = 'yolo_boxes_v4_0')(output_v4_0)
    boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name = 'yolo_boxes_v4_1')(output_v4_1)
    boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name = 'yolo_boxes_v4_2')(output_v4_2)
    outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name = 'yolo_nms')((boxes_v4_0[:3], boxes_v4_1[:3], boxes_v4_2[:3]))

    return Model(inputs, outputs_v4, name = 'yolov4')

def Darknet(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 32, 3)
    x = DarknetBlock(x, 64, 1)
    x = DarknetBlock(x, 128, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 256, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 512, 8)
    x = DarknetBlock(x, 1024, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None):
    x = inputs = Input([None, None, 3])
    x = DarknetConv(x, 16, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 32, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 64, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 128, 3)
    x = MaxPool2D(2, 2, 'same')(x)
    x = x_8 = DarknetConv(x, 256, 3)  # skip connection
    x = MaxPool2D(2, 2, 'same')(x)
    x = DarknetConv(x, 512, 3)
    x = MaxPool2D(2, 1, 'same')(x)
    x = DarknetConv(x, 1024, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)
    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)
    return yolo_conv

'''
def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output

def Yolov4Output(filters, anchors, classes, name=None): 
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters * 2, 3, batch_norm="LeakyReLU")
        x = DarknetConv(x, anchors * (classes + 5), 1, activation="Linear", batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)
    return yolo_output
#x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm= "Linear")
        #x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)


def yolo_boxes(pred, anchors, classes):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:3]
    box_xy, box_wh, objectness, class_probs = tf.split(
        pred, (2, 2, 1, classes), axis=-1)

    box_xy = tf.sigmoid(box_xy)
    objectness = tf.sigmoid(objectness)
    class_probs = tf.sigmoid(class_probs)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / \
        tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def yolo_nms(outputs, anchors, masks, classes):
    # boxes, conf, type
    b, c, t = [], [], []

    for o in outputs:
        b.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        c.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
        t.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

    bbox = tf.concat(b, axis=1)
    confidence = tf.concat(c, axis=1)
    class_probs = tf.concat(t, axis=1)

    scores = confidence * class_probs
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        scores=tf.reshape(
            scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=yolo_max_boxes,
        max_total_size=yolo_max_boxes,
        iou_threshold=yolo_iou_threshold,
        score_threshold=yolo_score_threshold
    )

    return boxes, scores, classes, valid_detections

'''
def YoloV3(size=None, channels=3, anchors=yolo_anchors,
           masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_36, x_61, x = Darknet(name='yolo_darknet')(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                     name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None, channels=3, anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels], name='input')

    x_8, x = DarknetTiny(name='yolo_darknet')(x)

    x = YoloConvTiny(256, name='yolo_conv_0')(x)
    output_0 = YoloOutput(256, len(masks[0]), classes, name='yolo_output_0')(x)

    x = YoloConvTiny(128, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(128, len(masks[1]), classes, name='yolo_output_1')(x)

    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                     name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                     name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')
'''
def YoloLoss(anchors, classes=80, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(
            y_pred, anchors, classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]


        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_true)[1]
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - \
            tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.zeros_like(true_wh), true_wh)


        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        best_iou = tf.map_fn(
            lambda x: tf.reduce_max(broadcast_diou(x[0], tf.boolean_mask(
                x[1], tf.cast(x[2], tf.bool))), axis=-1),
            (pred_box, true_box, obj_mask),
            tf.float32)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)


        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * \
            tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + \
            (1 - obj_mask) * ignore_mask * obj_loss
        # TODO: use binary_crossentropy instead
        class_loss = obj_mask * sparse_categorical_crossentropy(
            true_class_idx, pred_class)


        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))


        return xy_loss + wh_loss + obj_loss + class_loss
    return yolo_loss

