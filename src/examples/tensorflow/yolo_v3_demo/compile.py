import sys
import json
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# launch predictor and run inference on an arbitrary image in the validation dataset
yolo_pred_cpu = tf.contrib.predictor.from_saved_model('./yolo_v3_coco_saved_model')
image_path = './val2017/000000581781.jpg'
with open(image_path, 'rb') as f:
    feeds = {'image': [f.read()]}
results = yolo_pred_cpu(feeds)

# load annotations to decode classification result
with open('./annotations/instances_val2017.json') as f:
    annotate_json = json.load(f)
label_info = {idx+1: cat['name'] for idx, cat in enumerate(annotate_json['categories'])}

# draw picture and bounding boxes
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(Image.open(image_path).convert('RGB'))
wanted = results['scores'][0] > 0.1
for xyxy, label_no_bg in zip(results['boxes'][0][wanted], results['classes'][0][wanted]):
    xywh = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
    rect = patches.Rectangle((xywh[0], xywh[1]), xywh[2], xywh[3], linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    rx, ry = rect.get_xy()
    rx = rx + rect.get_width() / 2.0
    ax.annotate(label_info[label_no_bg + 1], (rx, ry), color='w', backgroundcolor='g', fontsize=10,
                ha='center', va='center', bbox=dict(boxstyle='square,pad=0.01', fc='g', ec='none', alpha=0.5))
plt.show()

import shutil
import tensorflow as tf
import tensorflow.neuron as tfn


def no_fuse_condition(op):
    return op.name.startswith('Preprocessor') or op.name.startswith('Postprocessor')

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], './yolo_v3_coco_saved_model')
    no_fuse_ops = [op.name for op in sess.graph.get_operations() if no_fuse_condition(op)]
shutil.rmtree('./yolo_v3_coco_saved_model_neuron', ignore_errors=True)
result = tfn.saved_model.compile(
    './yolo_v3_coco_saved_model', './yolo_v3_coco_saved_model_neuron',
    # to enforce trivial compilable subgraphs to run on CPU
    no_fuse_ops=no_fuse_ops,
    minimum_segment_size=100,
    batch_size=2,
    dynamic_batch_size=True,
)
print(result)