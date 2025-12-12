import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense





def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    epsilon = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    return mse, rmse, mae, mape


def plot_single_ticker_prediction(ticker, model, window_size):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()

    scaler = MinMaxScaler()
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

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_inv = scaler.inverse_transform(predictions)

    mse, rmse, mae, mape = calculate_metrics(y_test_inv, predictions_inv)

    print(f"\nМетрики для {ticker}:")
    print(f"MSE={mse:.4f} RMSE={rmse:.4f} MAE={mae:.4f} MAPE={mape:.2f}%")

    dates = df.index[window_size + split:]

    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_test_inv, label="Реальные значения")
    plt.plot(dates, predictions_inv, label="Прогноз LSTM")
    plt.title(f"{ticker}: прогноз цены акции")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def plot_residuals(y_true, y_pred):
    residuals = y_true.flatten() - y_pred.flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(residuals)
    plt.title("Ошибки прогноза (Residuals)")
    plt.xlabel("Наблюдение")
    plt.ylabel("Ошибка")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def plot_training_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title("Кривая обучения LSTM")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)


def remove_outliers_iqr(series):
    q1 = series.quantile(0.01)
    q99 = series.quantile(0.99)
    return series.clip(lower=q1, upper=q99)


def prepare_ticker_data(ticker, window_size):
    df = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df = df[['Close']].dropna()

    # обработка выбросов
    df['Close'] = remove_outliers_iqr(df['Close'])

    # масштабирование
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    # временные окна
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i, 0])
        y.append(scaled[i, 0])

    return np.array(X).reshape(-1, window_size, 1), np.array(y)




TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "INTC", "AMD",
    "NFLX", "ORCL", "JPM", "BAC",
    "KO", "PEP", "WMT", "COST", "DIS", "IBM"
]

START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
WINDOW_SIZE = 60
EPOCHS = 6
BATCH_SIZE = 32


X_all, y_all = [], []

for t in TICKERS:
    X_t, y_t = prepare_ticker_data(t, WINDOW_SIZE)
    X_all.append(X_t)
    y_all.append(y_t)

X_all = np.vstack(X_all)
y_all = np.concatenate(y_all)

split = int(0.8 * len(X_all))
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]



model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

predictions = model.predict(X_test)

mse, rmse, mae, mape = calculate_metrics(y_test, predictions)


plot_training_history(history)
plot_residuals(y_test, predictions)
plot_single_ticker_prediction("AAPL", model, WINDOW_SIZE)


plt.show()
