import torch
import torch.nn as nn
import pandas as pd
import gradio as gr
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence


df = pd.read_csv(r"C:\Users\sinha\Documents\Research Paper Projects\Sentament Analysis\twitter_sentiment.csv", names=['id', 'entity', 'sentiment', 'text'], on_bad_lines='skip')
df.head()

# 3. Define Model Architecture (Match Your Training Code)

class SentimentModel(nn.Module):
    def __init__(self, vocab_size=26304, embed_dim=128, hidden_dim=256, output_dim=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        return self.fc(hidden[-1])

# 4. Load Trained Weights

model = SentimentModel(vocab_size=26304, embed_dim=128, hidden_dim=256, output_dim=4, num_layers=2)
model.load_state_dict(torch.load(r"C:\Users\sinha\Documents\Research Paper Projects\Sentament Analysis\sentiment_model_weights.pth", map_location="cpu"))
model.eval()


# 5. Preprocessing: Tokenizer + Numericalization

tokenizer = get_tokenizer("basic_english")

# Simple static vocabulary (fast and lightweight)
# Adjust according to your training code if needed
from collections import Counter
counter = Counter()

for text in df['text'].astype(str):
    counter.update(tokenizer(text))

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(26304 - 2))}
vocab["<pad>"] = 0
vocab["<unk>"] = 1

def encode_text(text):
    tokens = tokenizer(text)
    if not tokens:
        return torch.tensor([0], dtype=torch.long)
    ids = [vocab.get(t, 1) for t in tokens]
    return torch.tensor(ids, dtype=torch.long)

def collate_batch(text):
    ids = encode_text(text)
    return pad_sequence([ids], batch_first=True, padding_value=0)

# 6. Prediction Function

# 6. Advanced Prediction & Explanation Functions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime.lime_text import LimeTextExplainer
import re
import torch.nn.functional as F

label_map = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
    3: "Irrelevant"
}

# Inverse map for LIME
inv_label_map = {v: k for k, v in label_map.items()}
class_names = [label_map[i] for i in sorted(label_map.keys())]

def predict_proba_for_lime(texts):
    """
    Prediction function for LIME.
    Input: list of strings
    Output: numpy array of shape (n_samples, n_classes) with probabilities
    """
    model.eval()
    probs_list = []
    
    # Process each text individually (or batch if optimized, but loop is safer for variable lengths here)
    for text in texts:
        # Encode and pad
        ids = encode_text(text)
        # Add batch dimension: [1, seq_len]
        x = ids.unsqueeze(0) 
        
        with torch.no_grad():
            output = model(x)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            probs_list.append(probs)
            
    return np.array(probs_list)

explainer = LimeTextExplainer(class_names=class_names)

def generate_explanation_plot(explanation):
    """
    Generate a matplotlib figure for the LIME explanation weights.
    """
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    return fig

def plot_probabilities(probabilities):
    """
    Plot bar chart of class probabilities.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=class_names, y=probabilities, ax=ax, palette='viridis')
    ax.set_title('Sentiment Probability Distribution')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    for i, p in enumerate(probabilities):
        ax.text(i, p + 0.01, f"{p:.2f}", ha='center')
    plt.tight_layout()
    return fig

def analyze_sentiment(text):
    if not text.strip():
        return "Please enter text.", None, None, None

    # 1. Get Prediction and Probabilities
    probs = predict_proba_for_lime([text])[0]
    pred_idx = np.argmax(probs)
    pred_label = label_map[pred_idx]
    
    # 2. Generate LIME Explanation
    exp = explainer.explain_instance(text, predict_proba_for_lime, num_features=10)
    
    # 3. Generate Plots
    prob_plot = plot_probabilities(probs)
    lime_plot = generate_explanation_plot(exp)
    
    # 4. Highlighted Text (HTML)
    # LIME's as_html() returns a full HTML string, we can extract the relevant part or use it as is.
    # However, for better integration, let's use the user's requested style or LIME's default.
    # LIME's default as_html() is good.
    lime_html = exp.as_html()
    
    return pred_label, prob_plot, lime_plot, lime_html

# 7. Gradio Interface

ui = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a tweet...", label="Input Tweet"),
    outputs=[
        gr.Textbox(label="Predicted Sentiment"),
        gr.Plot(label="Probability Distribution"),
        gr.Plot(label="LIME Feature Importance (Affected Words)"),
        gr.HTML(label="Detailed Explanation")
    ],
    title="Advanced Twitter Sentiment Classifier",
    description="Predict sentiment and visualize which words contributed most to the decision using LIME."
)

ui.launch()
