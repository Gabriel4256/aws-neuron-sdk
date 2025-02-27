from transformers import pipeline
import tensorflow as tf
import tensorflow.neuron as tfn

#Create the huggingface pipeline for sentiment analysis
#this model tries to determine of the input text has a positive
#or a negative sentiment.
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

pipe = pipeline('sentiment-analysis', model=model_name, framework='tf')

#pipelines are extremely easy to use as they do all the tokenization,
#inference and output interpretation for you.
pipe(['I love pipelines, they are very easy to use!', 'this string makes it batch size two'])

neuron_pipe = pipeline('sentiment-analysis', model=model_name, framework='tf')

#the first step is to modify the underlying tokenizer to create a static 
#input shape as inferentia does not work with dynamic input shapes
original_tokenizer = pipe.tokenizer


#you intercept the function call to the original tokenizer
#and inject our own code to modify the arguments
def wrapper_function(*args, **kwargs):
    kwargs['padding'] = 'max_length'
    #this is the key line here to set a static input shape
    #so that all inputs are set to a len of 128
    kwargs['max_length'] = 128 
    kwargs['truncation'] = True
    kwargs['return_tensors'] = 'tf'
    return original_tokenizer(*args, **kwargs)

#insert our wrapper function as the new tokenizer as well 
#as reinserting back some attribute information that was lost
#when you replaced the original tokenizer with our wrapper function
neuron_pipe.tokenizer = wrapper_function
neuron_pipe.tokenizer.decode = original_tokenizer.decode
neuron_pipe.tokenizer.mask_token_id = original_tokenizer.mask_token_id
neuron_pipe.tokenizer.pad_token_id = original_tokenizer.pad_token_id
neuron_pipe.tokenizer.convert_ids_to_tokens = original_tokenizer.convert_ids_to_tokens


#Now that our neuron_classifier is ready you can use it to
#generate an example input which is needed to compile the model
#note that pipe.model is the actual underlying model itself which 
#is what Tensorflow Neuron actually compiles.
from datasets import load_dataset
dataset = load_dataset('amazon_polarity')

string_inputs = dataset['test'][:128]['content']

example_inputs = neuron_pipe.tokenizer(string_inputs)
#compile the model by calling tfn.trace by passing in the underlying model
#and the example inputs generated by our updated tokenizer
neuron_model = tfn.trace(pipe.model, example_inputs)

#now you can insert the neuron_model and replace the cpu model
#so now you have a huggingface pipeline that uses and underlying neuron model!
neuron_pipe.model = neuron_model
neuron_pipe.model.config = pipe.model.config

#directly call the model
print(neuron_model(example_inputs))
#with the model inserted to the wrapper
print(neuron_pipe(string_inputs))

#Look at the difference between string_inputs
#and example_inputs

print(example_inputs)
print(string_inputs)

class TFBertForSequenceClassificationFlatIO(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.model({'input_ids': input_ids, 'attention_mask': attention_mask})
        return output['logits']

#wrap the original model from HuggingFace, now our model accepts a list as input
model_wrapped = TFBertForSequenceClassificationFlatIO(pipe.model)
#turn the dictionary input into list input
example_inputs_list = [example_inputs['input_ids'], example_inputs['attention_mask']]

#compile the wrapped model and save it to disk
model_wrapped_traced = tfn.trace(model_wrapped, example_inputs_list)
model_wrapped_traced.save('./distilbert_b128')


