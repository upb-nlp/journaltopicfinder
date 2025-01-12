from sentence_transformers import SentenceTransformer
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from bertopic import BERTopic


input_file = "ixdea_parsed.json"

with open(input_file, 'r') as openfile:
    all_articles = json.load(openfile)

# prepare data
input_data = []
article_dates = []
use_articles = False
nr_of_articles_per_year = [0] * 20
for article in all_articles['corpus']:
    if article['filename'][0:3] == "15_" and use_articles == False:
        use_articles = True
    if use_articles:
        nr_of_articles_per_year[(int(article['filename'][0:2]) - 15) // 4] += 1
        for sentence in article['text']:
            input_data.append(sentence)
            article_dates.append( (int(article['filename'][0:2]) - 15) // 4 + 2013)
print(nr_of_articles_per_year)
# load model
topic_model = BERTopic.load("saved_model_llama3")
print(topic_model.get_topic_info())
# # llama labels
llama_topic_labels = {topic: values[0][0].replace('"', '') for topic, values in topic_model.topic_aspects_["Llama"].items()}
llama_topic_labels[-1] = "Outlier Topic"
topic_model.set_topic_labels(llama_topic_labels)

# # topics map
fig1 = topic_model.visualize_topics(topics=[4, 6, 13, 24, 26, 40, 47, 78, 83, 91], custom_labels=True)
fig1.write_html("ixdea_topics_llama_top10.html")
fig1.write_image("topics_data_map.png")

# # display topics
print(topic_model.get_topic_info()[1:11])
for topic in range(11):
    print(topic)
    print(topic_model.get_topic(topic))

# # topics over time
print("toics over time")
topics_over_time = topic_model.topics_over_time(input_data, article_dates, nr_bins=20) 
fig2 = topic_model.visualize_topics_over_time(topics_over_time, topics=[4, 6, 13, 24, 26, 40, 47, 78, 83, 91], custom_labels=True)
fig2.write_html("ixdea_topics_over_time_llama_top10.html")

# # top 12 based on top 10 words
import plotly.express as px
fig4 = topic_model.visualize_barchart(
    topics=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13],
    n_words = 10, 
    custom_labels=True, 
    width=300,
    height=300,
    autoscale=True
)
# # Top 12 topics visualized by the frequency of the top 10 words
fig4.update_layout(
    margin=dict(l=20, r=50, t=80, b=20),

    plot_bgcolor='rgba(0,0,0,0)',

    title={
        'font': dict(
            family="Roboto Black",
            size=1,
            color="#000000"
        )
    },
    font=dict(
        family="Roboto",
        size=10,
        color="#000000"
    )
)

color_sequence = px.colors.qualitative.Vivid  # Choose a color sequence.
fig4.update_traces(marker_color=color_sequence)

# # Show the updated figure
fig4.write_image(file='fig4_ixdea_topics_llama_top12.png', format='png')

# # Show all representative documents for top 5 topics
for i in range(5):
    T = topic_model.get_document_info(input_data)
    docs_per_topics = T.groupby(["Topic"]).apply(lambda x: x.index).to_dict()
    docs = T.Document[docs_per_topics[i]] # list of documents for topic i

    articles = set()
    for article in all_articles['corpus']:
        for sentence in article['text']:
            if(sentence in docs.values):
                articles.add(article['filename'])
    with open('topics_related_articles.txt', 'a') as f:
        f.write(str(sorted(articles)) + '\n\n')