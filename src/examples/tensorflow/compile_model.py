import tensorflow as tf
import tensorflow.neuron as tfn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_saved_model', required=True, help='Original SaveModel')
parser.add_argument('-o', '--output_saved_model', required=True, help='Output SavedModel that runs on Inferentia')

args = parser.parse_args()

MODEL_DIR = args.input_saved_model

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], MODEL_DIR)
    result = tfn.saved_model.compile(
        args.input_saved_model, args.output_saved_model,
    )