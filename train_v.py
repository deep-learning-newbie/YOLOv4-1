import tensorflow as tf

from tensorflow.keras.losses import (
    binary_crossentropy,
    sparse_categorical_crossentropy
)

from tensorflow.keras.layers import (
  Lambda,
  Input,
)

from yolov4.models_v import (
  Mish,
  yolo_anchors_608,
  yolo_anchor_masks,
  yolo_boxes,
  yolo_nms,
  load_tfrecord_dataset,
  load_fake_dataset,
  transform_targets,
  transform_images,
  YoloLoss,
  make_yolov4_model,
  WeightReader,
)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)
size = 608
dataset = './data/voc2012_train.tfrecord'
v_dataset = './data/voc2012_val.tfrecord'
label_path = './data/voc2012.names'
#label_path  = './data/coco.names'
weights_path = './data/yolov4.weights'

labels = [c.strip() for c in open(label_path).readlines()]
transfer = None
transfer = 80
#num_classes = len(labels)
num_classes = transfer or len(labels)
learning_rate = 1e-4
epochs = 100
batch_size = 4
channels = 3

anchors = yolo_anchors_608
anchor_masks = yolo_anchor_masks
masks = yolo_anchor_masks

###./checkpoints/yolov4_test_u.h5  --> on VOC2012
'''
#model = YoloV4(size, training=True, classes=num_classes)
#model = tf.keras.models.load_model('./checkpoints/yolov4_train_test_E.h5',custom_objects={'Mish':Mish}, compile=False)
#pre_model = tf.keras.models.load_model('./checkpoints/yolov4.h5',compile = False)#custom_objects={'Mish':Mish}, compile=False)
#pre_model = tf.keras.models.load_model('./checkpoints/yolov4_test_u.h5',custom_objects = {'Mish':Mish}, compile = False)
'''
### 
pre_model = make_yolov4_model(size, classes_nb=num_classes)#training=True, classes=num_classes)
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(pre_model)

fine_tune_at = "convn_136"
train = True

for i in pre_model.layers:
  #if i.name == fine_tune_at:
  #  train = True
  i.trainable = train

x = inputs = Input([size, size, channels], name='inputs')
output_v4_0, output_v4_1, output_v4_2 = pre_model(x)
x = output_v4_0 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[0]), num_classes + 5)), name ='output_v4_0_137')(output_v4_0)
x = output_v4_1 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[1]), num_classes + 5)), name = 'output_v4_1_147')(output_v4_1)
x = output_v4_2 = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], len(masks[2]), num_classes + 5)), name = 'output_v4_2_157')(output_v4_2)

#boxes_v4_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes), name = 'yolo_boxes_v4_0')(output_v4_0)
#boxes_v4_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes), name = 'yolo_boxes_v4_1')(output_v4_1)
#boxes_v4_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes), name = 'yolo_boxes_v4_2')(output_v4_2)
#outputs_v4 = Lambda(lambda x: yolo_nms(x, anchors, masks, classes), name = 'yolo_nms')((boxes_v4_2[:3], boxes_v4_1[:3], boxes_v4_0[:3]))
model = tf.keras.models.Model(inputs = [inputs], outputs = [output_v4_2, output_v4_1, output_v4_0], name = 'Yolov4')
#model = Model(inputs = inputs, outputs = [output_v4_0, output_v4_1, output_v4_2], name = 'Yolov4')


train_dataset = load_fake_dataset()
train_dataset = load_tfrecord_dataset(dataset, label_path, size)
train_dataset = train_dataset.shuffle(buffer_size=512)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.map(lambda x, y: (
    transform_images(x, size),
    transform_targets(y, anchors, anchor_masks, size)))
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

a = train_dataset.take(2)
print(a)
val_dataset = load_fake_dataset()
val_dataset = load_tfrecord_dataset(v_dataset, label_path, size)

val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.map(lambda x, y: (
    transform_images(x, size),
    transform_targets(y, anchors, anchor_masks, size)))

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
loss = [YoloLoss(anchors[mask], classes=num_classes) for mask in anchor_masks]
model.compile(optimizer=optimizer, loss=loss, run_eagerly = True)#metrics = ['accuracy'])#run_eagerly=(FLAGS.mode == 'eager_fit'))
callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('./checkpoints/yolov4_voc.h5', verbose=1, save_best_only=True),
            TensorBoard(log_dir='logs', histogram_freq = 1, update_freq = 'epoch')
        ]
        ###'./checkpoints/yolov4_train_mish_lambda_E.h5' --> save model in condition using class Mish
        ### ./checkpoints/yolov4_train_mish_diou.h5' --> save model in condition using class Mish and dIoU
        ### './checkpoints/yolov4_test.h5' --> save model for testing
        ### './checkpoints/yolov4_test_u.h5' --> save model in condtion using voc2012.names on coco.names
        ### './checkpoints/yolov4_weight_histogram.h5' --> for histogram of weights
        ### './checkpoints/yolov4_2.h5' --> save model in condtion using yolov4_2

model.summary()
history = model.fit(train_dataset, epochs=epochs, callbacks=callbacks, validation_data=val_dataset)
