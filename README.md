# 📈 Stock Price Prediction with LSTM and Transformer Models

This repository contains two deep learning models implemented in PyTorch to predict stock prices based on historical time series data:
- 📘 **LSTM (Long Short-Term Memory) Model** in a Jupyter notebook
- 📙 **Transformer Model** in a Python script

Both models aim to learn temporal dependencies in stock price data to forecast future prices.

---

## 🔧 Project Structure

```
.
├── LSTM_model_stocks.ipynb                # LSTM-based stock price prediction (Jupyter Notebook)
├── Transformer_stock_price_prediction.py  # Transformer-based stock prediction model
├── GOOG.csv                               # Input dataset (Google stock prices)
```

---

## 📊 Dataset

- **Source**: `GOOG.csv`  
- **Feature Used**: Closing price of Google stocks  
- **Preprocessing**: The closing prices are normalized using `MinMaxScaler` before model training.

---

## 🧠 Models

### 1. LSTM Model (`LSTM_model_stocks.ipynb`)
- Uses LSTM layers to capture long-term dependencies.
- Trained on sequences of closing prices.
- Outputs predictions for the next time step.
- Visualizes actual vs. predicted prices.

### 2. Transformer Model (`Transformer_stock_price_prediction.py`)
- Implements a Transformer architecture with configurable layers and attention heads.
- Learns from sliding windows of past stock prices.
- Uses a custom `StockDataset` class for sequence generation.
- Trains using MSE loss and plots predicted vs. actual prices after training.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
```

### 2. Install Dependencies

```bash
pip install torch pandas matplotlib scikit-learn
```

### 3. Prepare Data

Ensure the `GOOG.csv` file is in the same directory. It should contain at least a `Close` column with daily closing prices of Google stock.

### 4. Run Models

- **LSTM Model**:  
  Open `LSTM_model_stocks.ipynb` in Jupyter and run all cells.

- **Transformer Model**:  
  Run the script:
  ```bash
  python Transformer_stock_price_prediction.py
  ```

---

## 📈 Output

Both models will generate plots comparing **actual** and **predicted** stock prices across time.

---

## 📌 Notes

- You can tune hyperparameters like `SEQ_LEN`, model dimensions, and number of epochs to improve performance.
- This is a simplified educational project; real-world forecasting requires more features and validation strategies.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
