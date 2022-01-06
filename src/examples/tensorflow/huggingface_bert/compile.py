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

#Create the huggingface pipeline for sentiment analysis
#this model tries to determine of the input text has a positive
#or a negative sentiment.
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

pipe = pipeline('sentiment-analysis', model=model_name, framework='tf')

#pipelines are extremely easy to use as they do all the tokenization,
#inference and output interpretation for you.
pipe(['I love pipelines, they are very easy to use!', 'this string makes it batch size two'])

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