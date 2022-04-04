import tensorflow as tf
# variable define ...
saver = tf.train.Saver()
with tf.Session() as sess:
  # Initialize v1 since the saver will not.
    saver.restore(sess, "/home/ubuntu/models/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt")
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    tf.saved_model.simple_save(
      sess,
      "./savedmodel/",
      inputs={"image": tensor_info_x},
      outputs={"scores": tensor_info_y}
    )