import os
import tensorflow as tf
import sys

trained_checkpoint_prefix = sys.argv[1]
export_dir = sys.argv[2]
# ex: './wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt'
export_dir = os.path.join(export_dir, 'saved_model')

graph = tf.Graph()
with tf.compat.v1.Session(graph=graph) as sess:
    # Restore from checkpoint
    loader = tf.compat.v1.train.import_meta_graph(trained_checkpoint_prefix + '.meta')
    loader.restore(sess, trained_checkpoint_prefix)

    # Export checkpoint to SavedModel
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)
    builder.add_meta_graph_and_variables(sess,
                                         [tf.saved_model.SERVING],
                                         strip_default_attrs=True)
    builder.save()    