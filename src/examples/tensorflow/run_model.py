import tensorflow_neuron.python.saved_model as saved_model
import numpy as np
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer, option_builder

# MODEL_DIR='./yolo_v3_demo/yolo_v3_coco_saved_model_neuron'
# MODEL_DIR = '/home/ubuntu/models/wwm_uncased_L-24_H-1024_A-16_neuron'
MODEL_DIR = "./models/bert/bert-2_neuron"
# MODEL_DIR = "./saved_model_test"

# def get_image_as_bytes(images, eval_pre_path):
#     batch_im_id_list = []
#     batch_im_name_list = []
#     batch_img_bytes_list = []
#     n = len(images)
#     batch_im_id = []
#     batch_im_name = []
#     batch_img_bytes = []
#     for i, im in enumerate(images):
#         im_id = im['id']
#         file_name = im['file_name']
#         if i % eval_batch_size == 0 and i != 0:
#             batch_im_id_list.append(batch_im_id)
#             batch_im_name_list.append(batch_im_name)
#             batch_img_bytes_list.append(batch_img_bytes)
#             batch_im_id = []
#             batch_im_name = []
#             batch_img_bytes = []
#         batch_im_id.append(im_id)
#         batch_im_name.append(file_name)

#         with open(os.path.join(eval_pre_path, file_name), 'rb') as f:
#             batch_img_bytes.append(f.read())
#     return batch_im_id_list, batch_im_name_list, batch_img_bytes_list


# image_names = ['000000000139.jpg']
# pre_path = './yolo_v3_demo/val2017'

# image = get_image_as_bytes(image_names, pre_path)

# model_feed_dict = {'image': np.array(image)}

## for bert-large
model_feed_dict = {
    'segment_ids': np.zeros((1,128)), 
    'input_ids': np.zeros((1,128)),
    'input_mask': np.zeros((1,128)), 
    'label_ids': [1],
    'is_real_example': [1],
}

# pred = tf.contrib.predictor.from_saved_model(MODEL_DIR)
# print(pred(model_feed_dict))

# options = option_builder.ProfileOptionBuilder.time_and_memory()
# options['select'] = ['micros', 'bytes', 'device', 'params']
# options['account_displayed_op_only'] = False
# print(options)

saved_model.profile(MODEL_DIR, model_feed_dict=model_feed_dict, timeline_json="./timeline.json", options=options)