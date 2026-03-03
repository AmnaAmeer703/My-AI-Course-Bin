import gradio as gr
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
def predict_sentiment(text):
    return classifier(text)[0]
gr.Interface(fn=predict_sentiment,
             inputs=gr.Textbox(label="Enter a sentence"),
             outputs=gr.Label(label="Sentiment"),
             title="Sentiment Classifier").launch()

