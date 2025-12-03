import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Налаштування стилю графіків
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# --- 1. ЗАВАНТАЖЕННЯ ДАНИХ ---
print("Завантаження даних...")
df = pd.read_csv('accident_clear_data.csv', sep=';')

# --- 2. ПІДГОТОВКА ДАНИХ ДЛЯ МОДЕЛІ ---

# Функція для створення спрощених категорій (для ML)
def clean_accident_cause_final(val):
    val = str(val).lower()
    if 'нетверезому' in val or 'алкоголь' in val: return 'Алкоголь'
    elif 'швидкост' in val: return 'Швидкість'
    elif 'пішохід' in val or 'переход' in val: return 'Пішохід'
    elif 'перехрест' in val or 'світлофор' in val: return 'Перехрестя'
    elif 'маневрування' in val: return 'Маневрування'
    elif 'дистанц' in val: return 'Дистанція'
    else: return 'Інше'

# Створюємо колонку Simple_Cause
if 'Simple_Cause' not in df.columns:
    df['Simple_Cause'] = df['mainAccidentCause'].apply(clean_accident_cause_final)

# Target: 1 = Тяжкі/Смертельні, 0 = Легкі
# Це наша цільова змінна
df['Is_Severe'] = (df['Count_Тяжко травмований'] + df['Count_Загинув']) > 0
df['Is_Severe'] = df['Is_Severe'].astype(int)

print(f"Розподіл наслідків (0 - легкі, 1 - тяжкі): \n{df['Is_Severe'].value_counts()}")

# --- 3. НАВЧАННЯ МОДЕЛІ (RANDOM FOREST) ---
print("\nНавчання моделі Random Forest...")

feature_cols = ['Hour', 'DayOfWeek', 'Month', 'district', 'Simple_Cause']
X = df[feature_cols].copy()
y = df['Is_Severe']

# Кодуємо текст у цифри
encoders = {}
for col in ['district', 'Simple_Cause']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# Тренуємо модель з балансуванням класів
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X, y)

# --- 4. ВІЗУАЛІЗАЦІЯ 1: Feature Importance ---
# Показує, ЯКІ фактори (час, місце, причина) впливають найбільше

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', legend=False, palette='viridis')
plt.title('Топ факторів, що впливають на тяжкість ДТП', fontsize=14)
plt.xlabel('Сила впливу')
plt.ylabel('')
plt.tight_layout()
plt.show()

# --- 5. ВІЗУАЛІЗАЦІЯ 2: Heatmap (Теплова карта) ---
# Показує, КОЛИ найнебезпечніше їздити

pivot_table = df.pivot_table(index='DayOfWeek', columns='Hour', values='Is_Severe', aggfunc='mean')
days_ua = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'П\'ятниця', 'Субота', 'Неділя']

plt.figure(figsize=(12, 5))
# Перевіряємо, чи індекс збігається з днями (0-6)
sns.heatmap(pivot_table, cmap='Reds', yticklabels=days_ua)
plt.title('Мапа смертельності: Де червоне — там найвища ймовірність тяжких травм', fontsize=14)
plt.xlabel('Година доби')
plt.ylabel('')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 6. ОБРОБКА МУЛЬТИ-ЛЕЙБЛІВ (EXPLODE) ---
# Це потрібно для третього графіка, щоб розділити "швидкість, алкоголь" на два записи

print("\nОбробка складних причин (розділення мульти-лейблів)...")
# Перетворюємо рядок "причина1, причина2" у список ["причина1", "причина2"]
df_exploded = df.copy()
df_exploded['mainAccidentCause'] = df_exploded['mainAccidentCause'].astype(str).apply(lambda x: x.split(', '))
# Розгортаємо список у рядки
df_exploded = df_exploded.explode('mainAccidentCause')
# Чистимо пробіли навколо тексту
df_exploded['mainAccidentCause'] = df_exploded['mainAccidentCause'].str.strip()

print(f"Записів до розділення: {len(df)}")
print(f"Записів після розділення: {len(df_exploded)}")

# --- 7. ВІЗУАЛІЗАЦІЯ 3: Детальний рейтинг причин ---

# Фільтруємо рідкісні причини (менше 20 випадків)
min_accidents = 20
cause_counts = df_exploded['mainAccidentCause'].value_counts()
common_causes = cause_counts[cause_counts >= min_accidents].index

df_filtered_causes = df_exploded[df_exploded['mainAccidentCause'].isin(common_causes)]

# Рахуємо відсоток тяжкості
severity_detailed = df_filtered_causes.groupby('mainAccidentCause')['Is_Severe'].mean().sort_values(ascending=False) * 100

plt.figure(figsize=(16, 10)) # Великий розмір для читабельності

# Кольори: Червоний (> середнього), Сірий (< середнього)
avg_severity = df['Is_Severe'].mean() * 100
colors = ['#d62728' if x > avg_severity else '#7f7f7f' for x in severity_detailed.values]

ax = sns.barplot(
    x=severity_detailed.values,
    y=severity_detailed.index,
    palette=colors,
    hue=severity_detailed.index,
    legend=False
)

plt.title(f'Топ найнебезпечніших причин ДТП (серед тих, що сталися >{min_accidents} разів)', fontsize=16)
plt.xlabel('Відсоток тяжких наслідків (%)', fontsize=12)
plt.ylabel('')

# Додаємо підписи відсотків
for i, v in enumerate(severity_detailed.values):
    ax.text(v + 0.2, i, f"{v:.1f}%", va='center', fontsize=10, fontweight='bold')

# Лінія середнього значення
plt.axvline(x=avg_severity, color='black', linestyle='--', alpha=0.5)
plt.text(avg_severity + 0.5, len(severity_detailed)-1, f'Середнє: {avg_severity:.1f}%', color='black')

plt.tight_layout()
plt.show()

# --- 8. КОНСОЛЬНИЙ ЗВІТ ---
stats = df_exploded.groupby('mainAccidentCause')['Is_Severe'].agg(['count', 'mean'])
stats = stats[stats['count'] >= min_accidents]
stats['severity_pct'] = stats['mean'] * 100
stats = stats.sort_values(by='severity_pct', ascending=False)

print(f"\n{'='*90}")
print(f" ДЕТАЛЬНИЙ РЕЙТИНГ НЕБЕЗПЕКИ (мін. {min_accidents} випадків)")
print(f"{'='*90}")
print(f"{'ПРИЧИНА':<60} | {'КІЛЬКІСТЬ':<10} | {'ТЯЖКІСТЬ (%)':<15}")
print(f"{'-'*60} | {'-'*10} | {'-'*15}")

for cause, row in stats.iterrows():
    print(f"{cause:<60} | {int(row['count']):<10} | {row['severity_pct']:.1f}%")

print(f"{'='*90}")