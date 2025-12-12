import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    epsilon = 1e-8  # защита от деления на 0
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return mse, rmse, mae, mape



def plot_single_ticker_prediction(ticker, model, window_size):
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )

    df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    split = int(0.8 * len(X))
    X_test = X[split:]
    y_test = y[split:]

    predictions = model.predict(X_test)

    predictions_inv = scaler.inverse_transform(predictions)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse, rmse, mae, mape = calculate_metrics(y_test_inv, predictions_inv)

    print(f"\nМетрики для акции {ticker}:")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE  = {mae:.4f}")
    print(f"MAPE = {mape:.2f}%")

    test_dates = df.index[window_size + split:]

    plt.figure(figsize=(14, 6))
    plt.plot(test_dates, y_test_inv, label="Реальные значения", linewidth=2)
    plt.plot(test_dates, predictions_inv, label="Прогноз LSTM", linewidth=2)

    plt.title(f"Прогноз цены акции {ticker} (multi-series LSTM)")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



def prepare_ticker_data(ticker, window_size):
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True
    )

    # убираем MultiIndex
    df.columns = df.columns.get_level_values(0)

    # берём только цену закрытия
    df = df[['Close']].dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)

    return X, y



TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "INTC", "AMD",
    "NFLX", "ORCL",
    "JPM", "BAC",
    "KO", "PEP",
    "WMT", "COST",
    "DIS",
    "IBM"
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
WINDOW_SIZE = 60
EPOCHS = 10
BATCH_SIZE = 32


X_all, y_all = [], []

for ticker in TICKERS:
    print(f"Загрузка данных: {ticker}")
    X_t, y_t = prepare_ticker_data(ticker, WINDOW_SIZE)
    X_all.append(X_t)
    y_all.append(y_t)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

print("Общий размер X:", X_all.shape)
print("Общий размер y:", y_all.shape)


# =========================
# TRAIN / TEST SPLIT (БЕЗ SHUFFLE)
# =========================
split = int(0.8 * len(X_all))

X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]


# =========================
# МОДЕЛЬ LSTM
# =========================
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# =========================
# ОБУЧЕНИЕ
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)


# =========================
# ОЦЕНКА КАЧЕСТВА (ГЛОБАЛЬНО)
# =========================
predictions = model.predict(X_test)

mse, rmse, mae, mape = calculate_metrics(y_test, predictions)

print("Глобальные метрики (multi-series model):")
print(f"MSE  = {mse:.6f}")
print(f"RMSE = {rmse:.6f}")
print(f"MAE  = {mae:.6f}")
print(f"MAPE = {mape:.2f}%")

