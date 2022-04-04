import tensorflow as tf
import numpy as np
import shutil

# variable define ...
# saver = tf.train.Saver()
with tf.Session(graph=tf.Graph()) as sess:
  # Initialize v1 since the saver will not.
    segment_ids = tf.saved_model.utils.build_tensor_info(tf.constant(np.zeros((1,128))))
    input_ids = tf.saved_model.utils.build_tensor_info(tf.constant(np.zeros((1,128))))
    input_mask = tf.saved_model.utils.build_tensor_info(tf.constant(np.zeros((1,128))))
    # label_ids = tf.saved_model.utils.build_tensor_info(tf.constant(np.zeros((1))))
    label = tf.saved_model.utils.build_tensor_info(tf.constant(np.zeros((1))))
    
    loader = tf.compat.v1.train.import_meta_graph('/home/ubuntu/models/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt.meta')
    loader.restore(sess, "/home/ubuntu/models/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt")
    shutil.rmtree("./saved_model_test", ignore_errors=True)
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder("./saved_model_test")


    signature_def = tf.compat.v1.saved_model.build_signature_def(
        inputs={
            "segment_ids": segment_ids,
            "input_ids": input_ids,
            "input_mask": input_mask,
        },
        outputs={"label": label})
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.SERVING],
                                         signature_def_map={
                                              tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def},
                                         strip_default_attrs=True)
    builder.save()    

    # segment_ids = 
    # tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    # tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    # tf.saved_model.simple_save(
    #   sess,
    #   "./savedmodel/",
    #   inputs={
    #       "segment_ids": segment_ids,
    #       "input_ids": input_ids,
    #       "input_mask": input_mask,
    #   },
    #   outputs={"label": label}
    # )