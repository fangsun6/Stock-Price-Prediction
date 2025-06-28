
# 📈 Stock Price Prediction with LSTM and Transformer Models

This project explores time series forecasting of stock prices using both a **recurrent neural network (LSTM)** and a **Transformer-based model**, trained on Google stock price data. The goal is to learn temporal dependencies from historical stock prices and forecast future values.

---

## 🎯 Project Objective

Stock price prediction is a classic problem in finance and machine learning. However, prices are noisy, influenced by multiple exogenous variables, and exhibit long-range dependencies. The purpose of this project is to:
- Benchmark **LSTM**, a standard choice for time series,
- Compare it with **Transformer**, a newer architecture designed for long-term dependency learning,
- Identify strengths and weaknesses of each model based on training dynamics, error metrics, and interpretability.

---

## 📦 Dataset

- Source: `GOOG.csv` containing historical **Google stock prices**.
- Feature used: **Closing price** (normalized using `MinMaxScaler`).
- Sequence length: `150` days.
- Data shape: one-dimensional univariate time series (Closing Price only).

---

## 🧠 Thought Process & Modeling Strategy

### Step 1: Start Simple with LSTM
- **Why**: LSTM captures temporal patterns via memory cells. It’s a traditional starting point for sequential data.
- **Architecture**: Multi-layer LSTM followed by a fully connected layer.
- **Target**: Next-day closing price based on previous 150 days.

### Step 2: Challenge the LSTM with a Transformer
- **Why**: Transformers can model long-range dependencies and parallelize training. But unlike NLP data, stock prices lack discrete tokens and positional embedding has to be carefully handled.
- **Approach**: Sequence-to-sequence prediction with the Transformer encoder-decoder using the same input as both source and target.

---

## 🔬 Technical Breakdown

### Preprocessing
- Normalized the data using `MinMaxScaler`.
- Split data into overlapping sequences of length 150.
- Used PyTorch’s `Dataset` and `DataLoader` for efficient batching.

### LSTM Model (see notebook)
- Implemented in `LSTM_model_stocks.ipynb`
- Sequential training using MSELoss
- Predictions stored and rescaled for plotting

### Transformer Model
- Implemented in `Transformer_stock_price_prediction.py`
- Used PyTorch’s `nn.Transformer` module
- Handled same input for source and target with MSE loss
- Fully connected layer mapped output to prediction window

---

## 🧩 Challenges & Solutions

| Challenge | Solution |
|----------|----------|
| **Stock prices are noisy** | Used long sequences to average out short-term noise |
| **LSTM overfitting** | Used dropout and regularization (if extended) |
| **Transformer struggles with short input** | Increased `d_model` and head count to better encode input |
| **Vanishing gradients** | Carefully initialized weights and used Adam optimizer |
| **Output misalignment** | Took last value of sequence for prediction and reshaped outputs to match actuals |
| **Evaluation bias due to scaling** | Inverted normalization before plotting results |
| **Slow training on long sequences** | Reduced batch size and used GPU acceleration |

---

## 📈 Results & Reflection

- **LSTM** performed smoothly, learning basic trends but occasionally lagging during sharp price transitions.
- **Transformer** was more powerful for trend detection but harder to train and more sensitive to hyperparameters.
- Visual comparison showed Transformer can overshoot but responds quicker to momentum changes.

---

## 📁 File Structure

```
.
├── LSTM_model_stocks.ipynb                # LSTM-based stock price prediction (Jupyter Notebook)
├── Transformer_stock_price_prediction.py  # Transformer-based stock prediction model
├── GOOG.csv                               # Input dataset (Google stock prices)
└── README.md                              # Project overview and reflection
```

---

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib torch scikit-learn
```

Then run:
- `LSTM_model_stocks.ipynb` for notebook-based LSTM model
- `Transformer_stock_price_prediction.py` for script-based transformer model

---

## 💡 Future Work

- Add attention visualization for Transformer interpretability.
- Compare with hybrid CNN-LSTM models.
- Incorporate more features (e.g., volume, open, high, low).
- Use rolling validation instead of one-time split.

---

## 📜 License

MIT License — for open research and education.

