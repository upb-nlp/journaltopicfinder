from sentence_transformers import SentenceTransformer
import json
import umap
import hdbscan
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from bertopic import BERTopic
# Load model directly
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from bertopic.representation import TextGeneration
import os
from huggingface_hub import hf_hub_download  # For loading the model instance from Hugging Face
# from llama_cpp import Llama  # LLM Wrapper
from bertopic.representation import LlamaCPP  # Representation Comparison


input_file = "ixdea_parsed.json"

with open(input_file, 'r') as openfile:
    all_articles = json.load(openfile)

# prepare data
input_data = []
use_articles = False
for article in all_articles['corpus']:
    # input_data.append(" ".join(article['text']))
    if article['filename'][0:3] == "15_" and use_articles == False:
        use_articles = True
    if use_articles:
        for sentence in article['text']:
            input_data.append(sentence)

print("total number of sentences = " + str(len(input_data)))
print("longest sentence (in words) = " + 
    str(max([len(article) for article in input_data])))
print("mean length of sentences (in words) = " + 
    str(sum([len(article) for article in input_data]) / len(input_data)))

# embedding data
model = SentenceTransformer('all-mpnet-base-v2')
#embeddings = model.encode(input_data, show_progress_bar=True)
#print("embedding size = " + str(len(embeddings[0])))

# dimensionality reduction
print("dim reduction")
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=50, 
                            metric='cosine') #.fit_transform(embeddings)

# clustering
print("clustering")
cluster = hdbscan.HDBSCAN(min_cluster_size=25,
                          metric='euclidean',                      
                          cluster_selection_method='leaf',
                          prediction_data=True) #.fit(umap_embeddings) # 'eom' genereaza doare 2 clustere


# count vectorizer 
count = CountVectorizer(ngram_range=(1, 2), stop_words="english") #.fit(docs_per_topic.Doc.values)

 
# System prompt describes information given to all conversations
system_prompt = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant for labeling topics.<|eot_id|>
Topics which are too general should be excluded. 
"""

# Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
main_prompt = """
<|start_header_id|>user<|end_header_id|>
I have a topic that  contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more. <|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

prompt = system_prompt + main_prompt

from bertopic.representation import TextGeneration
from bertopic import BERTopic

# topic Llama 3 representation
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token="")

# Llama 3 Model
llama = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map='auto', token="")
llama.eval()

# Our text generator
generator = transformers.pipeline(
    model=llama, tokenizer=tokenizer,
    task='text-generation',
    temperature=0.1,
    max_new_tokens=500,
    repetition_penalty=1.1
)

# Text generation with Llama 2
llama = TextGeneration(generator, prompt=prompt)
representation_model = {
    "Llama": llama,
}

print("before bertopic")
# Create our BERTopic model
topic_model = BERTopic(embedding_model=model, 
                       umap_model=umap_embeddings, 
                       hdbscan_model=cluster,
                       vectorizer_model=count,
                       representation_model=representation_model,
                       verbose=True)
print("before fit")
topic_model.fit_transform(input_data)
print(topic_model.get_topic_info())
print(topic_model.get_topic_info()["Llama"])
topic_model.save("saved_model_llama_test", 
                 serialization="safetensors", 
                 save_ctfidf=True, 
                 save_embedding_model=model)

