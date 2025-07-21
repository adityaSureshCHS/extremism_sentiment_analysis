#the code for all the hateBERT analysis

from transformers import pipeline


pipe = pipeline(model="GroNLP/hateBERT")


print(pipe("I hate everyone Indian"))