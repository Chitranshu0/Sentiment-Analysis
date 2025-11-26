# Sentiment Analysis of Short-Text Social Media Data Using Long Short-Term Memory (LSTM) Networks

## Abstract

This repository contains the implementation and supplementary materials for a study on sentiment classification of Twitter data. The project leverages deep learning techniques, specifically Long Short-Term Memory (LSTM) networks, to capture sequential dependencies in short-text data. Furthermore, the study incorporates Local Interpretable Model-agnostic Explanations (LIME) to provide interpretability for the model's predictions, bridging the gap between black-box neural network performance and human-understandable reasoning.

## 1. Introduction

Social media platforms generate vast amounts of unstructured textual data, presenting both a challenge and an opportunity for automated sentiment analysis. Traditional bag-of-words models often fail to capture the contextual nuances of language, such as sarcasm or negation. This project addresses these limitations by employing a Recurrent Neural Network (RNN) variant—the LSTM—which is designed to mitigate the vanishing gradient problem and effectively model long-range dependencies in text sequences.

## 2. Methodology

The core of the proposed system is a supervised learning pipeline that processes raw text, maps it to a high-dimensional vector space, and classifies it into one of four sentiment categories: Positive, Negative, Neutral, or Irrelevant.

### 2.1. Vector Space Representation (Word Embeddings)

To facilitate numerical computation, discrete textual tokens are mapped to continuous vector representations. Let $V$ be the vocabulary size and $d$ be the embedding dimension. Each word $w$ is represented as a vector $e_w \in \mathbb{R}^d$. This dense representation allows the model to capture semantic similarities between words based on their geometric proximity in the vector space.

### 2.2. Sequential Modeling with LSTM

The primary architectural component is the LSTM unit. Unlike standard RNNs, LSTMs maintain a cell state $C_t$ that acts as a conveyor belt for information, regulated by three distinct gates: the forget gate, the input gate, and the output gate.

For a given input sequence at time step $t$, denoted as $x_t$, and the previous hidden state $h_{t-1}$, the transition equations are defined as follows:

1.  **Forget Gate ($f_t$)**: Determines what information to discard from the cell state.
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$

2.  **Input Gate ($i_t$)**: Decides which new information is stored in the cell state.
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$

3.  **Cell State Update ($C_t$)**: The old cell state is updated by forgetting the irrelevant parts and adding the new candidate values.
    $$ C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t $$

4.  **Output Gate ($o_t$)** and **Hidden State ($h_t$)**: Computes the output based on the cell state and the filtered input.
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
    $$ h_t = o_t \odot \tanh(C_t) $$

*Where $\sigma$ denotes the sigmoid activation function, and $\odot$ represents element-wise multiplication.*

### 2.3. Classification Layer

The final hidden state $h_T$ of the sequence is passed through a fully connected (dense) layer. To obtain a probability distribution over the $K$ sentiment classes, the Softmax function is applied to the logits $z$:

$$ P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} $$

This yields the conditional probability that the input text $x$ belongs to class $j$.

### 2.4. Model Interpretability (LIME)

To ensure the reliability of the model, we employ LIME to approximate the complex non-linear decision boundary locally with an interpretable linear model. This allows us to identify specific tokens that exert the most significant influence on the prediction.

The explanation model $\xi(x)$ is obtained by minimizing the following objective:

$$ \xi(x) = \underset{g \in G}{\text{argmin}} \ \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

Here, $\mathcal{L}$ measures the fidelity of the explanation $g$ to the original model $f$ within the locality $\pi_x$, and $\Omega(g)$ penalizes the complexity of the explanation.

## 3. Implementation and Usage

The project is structured to allow for reproducibility and ease of testing.

### Prerequisites
*   Python 3.8+
*   PyTorch
*   Pandas, NumPy
*   Gradio (for the demonstration interface)

### Repository Structure
*   `UI.py`: The interactive application script utilizing Gradio for real-time inference.
*   `Research_Paper.ipynb`: The computational notebook detailing the data preprocessing, model training, and validation steps.
*   `sentiment_model_weights.pth`: The serialized parameters of the trained LSTM model.
*   `twitter_sentiment.csv`: The dataset used for training and evaluation.

### Execution
To launch the inference interface locally:

```bash
pip install -r requirements.txt  # Ensure dependencies are installed
python UI.py
```

## 4. Conclusion

This implementation demonstrates the efficacy of LSTM networks in handling the sequential nature of social media text. By integrating LIME, the system not only achieves high classification accuracy but also provides necessary transparency, making it suitable for applications requiring explainable AI.
