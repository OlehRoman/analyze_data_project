import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. ПІДГОТОВКА ДАНИХ ---
df = pd.read_csv('combined_accidents.csv', sep=';')

# Нам треба перетворити 'accidentDate' у формат datetime
# Спробуємо автоматично, якщо не вийде - треба вказати format (напр. '%d.%m.%Y' або '%Y-%m-%d')
df['Date'] = pd.to_datetime(df['accidentDate'])

# Prophet вимагає суворого формату:
# Колонка 'ds' (datestamp) - дата
# Колонка 'y' (value) - те, що прогнозуємо (кількість ДТП)

# Групуємо: рахуємо скільки аварій сталось кожного дня
daily_df = df.groupby('Date').size().reset_index(name='y')
daily_df.columns = ['ds', 'y']

print(f"Всього днів з даними: {len(daily_df)}")
print(daily_df.tail())

# --- 2. РОЗДІЛЕННЯ НА TRAIN / TEST ---
# Щоб перевірити точність, ми сховаємо від моделі останній місяць (30 днів)
# І попросимо спрогнозувати його, а потім звіримо з реальністю.

test_days = 30
train_df = daily_df.iloc[:-test_days]
test_df = daily_df.iloc[-test_days:]

print(f"\nНавчаємось на {len(train_df)} днях, тестуємо на останніх {len(test_df)} днях.")

# --- 3. НАВЧАННЯ PROPHET ---
# weekly_seasonality=True -> враховувати дні тижня (п'ятниця vs неділя)
# yearly_seasonality=True -> враховувати пори року (зима vs літо)
m = Prophet(weekly_seasonality=True, yearly_seasonality=True)

# Додаємо свята України (це фішка Prophet) - вони часто впливають на трафік
m.add_country_holidays(country_name='UA')

m.fit(train_df)

# --- 4. ПРОГНОЗУВАННЯ (EVALUATION) ---
# Робимо прогноз для дат з тестового набору
forecast = m.predict(test_df)

# Беремо прогнозовані значення ('yhat')
y_true = test_df['y'].values
y_pred = forecast['yhat'].values

# --- 5. МЕТРИКИ ТОЧНОСТІ ---
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"\n{'='*40}")
print(f"РЕЗУЛЬТАТИ ТОЧНОСТІ ПРОГНОЗУ:")
print(f"{'='*40}")
print(f"MAE (Середня помилка): {mae:.2f} ДТП")
print(f"RMSE (Квадратична помилка): {rmse:.2f}")
print(f"Середня кількість ДТП на день: {daily_df['y'].mean():.2f}")
print(f"Помилка у відсотках (приблизно): {(mae / daily_df['y'].mean()) * 100:.1f}%")
print(f"{'='*40}")

# --- 6. ПРОГНОЗ НА МАЙБУТНЄ (FUTURE) ---
# Тепер навчимо модель на ВСІХ даних і спрогнозуємо наступні 30 днів
m_full = Prophet(weekly_seasonality=True, yearly_seasonality=True)
m_full.add_country_holidays(country_name='UA')
m_full.fit(daily_df)

future = m_full.make_future_dataframe(periods=30) # +30 днів у майбутнє
forecast_full = m_full.predict(future)

# --- 7. ВІЗУАЛІЗАЦІЯ ---

# Графік 1: Прогноз
fig1 = m_full.plot(forecast_full)
plt.title('Прогноз кількості ДТП у Львові на наступні 30 днів')
plt.xlabel('Дата')
plt.ylabel('Кількість ДТП')
plt.show()

# Графік 2: Компоненти (Тренд, Тижневість, Річна сезонність)
fig2 = m_full.plot_components(forecast_full)
plt.show()