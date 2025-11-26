# 🧠 Advanced Twitter Sentiment Analysis

Welcome to the **Twitter Sentiment Analysis** project! This isn't just a regular code project; it's a deep dive into understanding human emotions using the power of **Artificial Intelligence** and **Mathematics**.

Imagine having a super-smart assistant that can read thousands of tweets in a second and tell you exactly how people are feeling—whether they are **Happy**, **Sad**, **Neutral**, or if the tweet is just **Irrelevant**. That's exactly what we built here!

---

## 🚀 How It Works (Simply Put)

1.  **Input**: You feed the model a tweet (like "I love this sunny day!").
2.  **Processing**: The computer breaks this sentence down into numbers (because computers love numbers, not words).
3.  **Thinking (The AI Brain)**: It uses a special brain structure called **LSTM** (Long Short-Term Memory) to understand the *context*. It knows that "not good" is different from "good" because it remembers the word "not" from before.
4.  **Decision**: It gives a score for each emotion and picks the highest one.
5.  **Explanation**: It even highlights *why* it picked that emotion (e.g., highlighting the word "love").

---

## 🧮 The "Super Math" Behind the Magic

To make this work, we use some really cool mathematical concepts. If you are writing a research paper, these formulas are the "secret sauce" of our model.

### 1. Word Embeddings (Turning Words into Math)
First, we turn every word into a list of numbers (a vector). Imagine a map where similar words like "happy" and "joy" are close together.
$$ E(w) \in \mathbb{R}^d $$
*Where $E(w)$ is the vector for word $w$, and $d$ is the size of that vector (we used 128 dimensions!).*

### 2. LSTM (The Memory Cell)
This is the heart of the project. Standard neural networks forget what they just read, but **LSTMs** remember! They have "gates" that decide what to keep and what to throw away.

Here are the actual equations the computer solves for every single word:

*   **Forget Gate ($f_t$)**: "Should I forget the previous stuff?"
    $$ f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f) $$
*   **Input Gate ($i_t$)**: "Is this new word important?"
    $$ i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i) $$
*   **Cell State ($C_t$)**: Updating the long-term memory.
    $$ \tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C) $$
    $$ C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t $$
*   **Output Gate ($o_t$)**: "What should I tell the next layer?"
    $$ o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) $$
    $$ h_t = o_t \cdot \tanh(C_t) $$

*Don't worry if it looks scary! It's just a fancy way of saying: "Take the old memory, mix it with the new word, and decide what to remember next."*

### 3. Softmax (Making the Final Decision)
Once the LSTM finishes reading the tweet, it gives us raw scores (logits). We use the **Softmax Function** to turn these scores into probabilities (percentages).

$$ P(y=j|x) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}} $$

*   $z_j$ is the score for emotion $j$.
*   The result is a probability between 0 and 1. If "Positive" gets 0.95, the model is 95% sure!

### 4. LIME (Explainability)
We don't just want the answer; we want to know *why*. We use **LIME** (Local Interpretable Model-agnostic Explanations). It tries to fit a simple math line locally around our specific tweet to see which words pushed the decision the most.

$$ \xi(x) = \underset{g \in G}{\text{argmin}} \ \mathcal{L}(f, g, \pi_x) + \Omega(g) $$

*In simple English: It tests variations of your sentence to see which words carry the most weight.*

---

## 🛠️ Project Structure

*   `UI.py`: The main application code. It runs the website where you can type tweets.
*   `sentiment_model_weights.pth`: The "brain" of the AI. This file contains the learned patterns from training.
*   `Research_Paper.ipynb`: The laboratory notebook where we trained and tested the model.
*   `twitter_sentiment.csv`: The dataset used to teach the AI.

## 💻 How to Run This

1.  **Install the requirements**:
    ```bash
    pip install torch pandas gradio torchtext matplotlib seaborn lime
    ```
2.  **Run the App**:
    ```bash
    python UI.py
    ```
3.  **Open your browser**: Click the link that appears (usually `http://127.0.0.1:7860`).

---

*Created with ❤️ for research and learning.*
