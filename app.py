import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- 1. НАЛАШТУВАННЯ СТОРІНКИ (Має бути першим) ---
st.set_page_config(page_title="Аналіз ДТП Львів", layout="wide", page_icon="🚗")


# --- 2. ФУНКЦІЇ ЗАВАНТАЖЕННЯ ДАНИХ ---
@st.cache_data
def load_main_data():
    df = pd.read_csv('accident_clear_data.csv', sep=';')

    def clean_accident_cause_final(val):
        val = str(val).lower()
        if 'нетверезому' in val or 'сп\'яніння' in val or 'алкоголь' in val:
            return 'Алкоголь'
        elif 'швидкост' in val:
            return 'Швидкість'
        elif 'пішохід' in val or 'переход' in val or 'невстановленому' in val:
            return 'Пішохід'
        elif 'перехрест' in val or 'пріоритет' in val or 'світлофор' in val:
            return 'Перехрестя/Світлофор'
        elif 'маневрування' in val or 'розворот' in val:
            return 'Маневрування'
        elif 'обгін' in val or 'зустрічн' in val:
            return 'Обгін/Зустрічка'
        elif 'дистанц' in val:
            return 'Дистанція'
        elif 'невідомо' in val:
            return 'Невідомо'
        else:
            return 'Інше'

    df['Simple_Cause'] = df['mainAccidentCause'].apply(clean_accident_cause_final)
    return df


@st.cache_data
def load_prophet_data():
    # Завантаження даних для 3-ї частини (Prophet)
    # Якщо файл той самий, можна використовувати load_main_data,
    # але у твоєму коді був 'combined_accidents.csv'
    try:
        df = pd.read_csv('combined_accidents.csv', sep=';')
    except FileNotFoundError:
        # Fallback якщо файли однакові
        df = pd.read_csv('accident_clear_data.csv', sep=';')
    return df


# --- 3. НАВІГАЦІЯ ---
st.sidebar.title("🗂️ Меню")
page = st.sidebar.radio("Оберіть розділ:",
                        ["🗺️ Інтерактивна карта", "📊 Аналіз факторів (ML)", "📈 Прогноз (Prophet)"])

# ==============================================================================
# СТОРІНКА 1: ІНТЕРАКТИВНА КАРТА (Твій перший файл)
# ==============================================================================
if page == "🗺️ Інтерактивна карта":
    st.title("🗺️ Інтерактивна карта ДТП у Львові")
    st.markdown("Кластеризація аварійно-небезпечних ділянок (DBSCAN).")

    df = load_main_data()

    st.sidebar.header("🔍 Фільтри карти")
    if 'Year' in df.columns:
        years = sorted(df['Year'].unique())
        selected_years = st.sidebar.slider("Роки", min(years), max(years), (min(years), max(years)))
        df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
    else:
        df_filtered = df.copy()

    hour_range = st.sidebar.slider("Час доби", 0, 23, (0, 23))
    df_filtered = df_filtered[(df_filtered['Hour'] >= hour_range[0]) & (df_filtered['Hour'] <= hour_range[1])]

    all_causes = sorted(df['Simple_Cause'].unique())
    selected_causes = st.sidebar.multiselect("Причини", all_causes, default=all_causes)
    df_filtered = df_filtered[df_filtered['Simple_Cause'].isin(selected_causes)]

    st.sidebar.markdown("---")
    eps_meters = st.sidebar.slider("Радіус (м)", 20, 200, 70)
    min_samples = st.sidebar.slider("Мін. аварій", 2, 20, 5)

    if len(df_filtered) > 0:
        coords = df_filtered[['latitude', 'longitude']].values
        coords_rad = np.radians(coords)
        kms_per_radian = 6371.0
        epsilon = (eps_meters / 1000) / kms_per_radian

        db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(coords_rad)
        df_filtered['Cluster'] = db.labels_

        total_accidents = len(df_filtered)
        noise_count = np.sum(df_filtered['Cluster'] == -1)
        clustered_count = total_accidents - noise_count
        noise_percent = (noise_count / total_accidents) * 100
        clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

        # Метрики
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Всього ДТП", total_accidents)
        c2.metric("Зон концентрації", clusters_found, f"{clustered_count} аварій")
        c3.metric("Шум (поодинокі)", f"{noise_percent:.1f}%")
        avg_acc = clustered_count / clusters_found if clusters_found > 0 else 0
        c4.metric("Сер. ДТП на зону", f"{avg_acc:.1f}")

        # Карта
        cluster_stats = df_filtered[df_filtered['Cluster'] != -1].groupby('Cluster').agg({
            'latitude': 'mean', 'longitude': 'mean',
            'Simple_Cause': lambda x: x.mode()[0] if not x.mode().empty else 'Інше',
            'accidentDay': 'count'
        }).rename(columns={'accidentDay': 'AccidentCount'})

        m = folium.Map(location=[49.8397, 24.0297], zoom_start=12)
        colors = {'Швидкість': 'red', 'Алкоголь': 'black', 'Перехрестя/Світлофор': 'orange',
                  'Пішохід': 'purple', 'Маневрування': 'blue', 'Обгін/Зустрічка': 'darkred',
                  'Дистанція': 'cadetblue', 'Невідомо': 'lightgray', 'Інше': 'green'}

        for cid, row in cluster_stats.iterrows():
            cause = row['Simple_Cause']
            color = colors.get(cause, 'gray')
            radius = min(6 + (np.log1p(row['AccidentCount']) * 4), 25)
            folium.CircleMarker(
                [row['latitude'], row['longitude']], radius=radius, color=color, fill=True, fill_color=color,
                fill_opacity=0.7,
                tooltip=f"{cause}: {row['AccidentCount']}"
            ).add_to(m)

        st_folium(m, width=1000, height=600)
        st.write("### Деталі по зонах")
        st.dataframe(cluster_stats.sort_values(by='AccidentCount', ascending=False), use_container_width=True)
    else:
        st.warning("Немає даних для відображення.")

# ==============================================================================
# СТОРІНКА 2: АНАЛІЗ ФАКТОРІВ (Твій другий файл)
# ==============================================================================
elif page == "📊 Аналіз факторів (ML)":
    st.title("📊 Аналіз факторів тяжкості ДТП")
    st.markdown("Використання **Random Forest** для визначення причин тяжких наслідків.")

    df = load_main_data()

    # Підготовка даних
    df['Is_Severe'] = (df['Count_Тяжко травмований'] + df['Count_Загинув']) > 0
    df['Is_Severe'] = df['Is_Severe'].astype(int)

    st.write("#### 1. Розподіл тяжкості")
    counts = df['Is_Severe'].value_counts()
    c1, c2 = st.columns(2)
    c1.metric("Легкі ДТП (0)", counts.get(0, 0))
    c2.metric("Тяжкі/Смертельні (1)", counts.get(1, 0))

    if st.button("🚀 Запустити навчання моделі"):
        with st.spinner('Навчання Random Forest...'):
            feature_cols = ['Hour', 'DayOfWeek', 'Month', 'district', 'Simple_Cause']
            X = df[feature_cols].copy()
            y = df['Is_Severe']

            encoders = {}
            for col in ['district', 'Simple_Cause']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            rf.fit(X, y)
        st.success("Модель навчено!")

        # Графік 1: Feature Importance
        st.subheader("Топ факторів впливу")
        importances = rf.feature_importances_
        fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance',
                                                                                            ascending=False)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis', ax=ax1)
        st.pyplot(fig1)

        # Графік 2: Heatmap
        st.subheader("Мапа небезпеки за часом")
        pivot = df.pivot_table(index='DayOfWeek', columns='Hour', values='Is_Severe', aggfunc='mean')
        days_ua = ['Понеділок', 'Вівторок', 'Середа', 'Четвер', 'П\'ятниця', 'Субота', 'Неділя']

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, cmap='Reds', yticklabels=days_ua, ax=ax2)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Детальний рейтинг небезпеки")

    # Мульти-лейбли
    df_exploded = df.copy()
    df_exploded['mainAccidentCause'] = df_exploded['mainAccidentCause'].astype(str).apply(lambda x: x.split(', '))
    df_exploded = df_exploded.explode('mainAccidentCause')
    df_exploded['mainAccidentCause'] = df_exploded['mainAccidentCause'].str.strip()

    min_accidents = 20
    stats = df_exploded.groupby('mainAccidentCause')['Is_Severe'].agg(['count', 'mean'])
    stats = stats[stats['count'] >= min_accidents]
    stats['severity_pct'] = stats['mean'] * 100
    stats = stats.sort_values(by='severity_pct', ascending=False)

    # Графік 3
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    avg_sev = df['Is_Severe'].mean() * 100
    colors = ['#d62728' if x > avg_sev else '#7f7f7f' for x in stats['severity_pct']]
    sns.barplot(x=stats['severity_pct'], y=stats.index, palette=colors, ax=ax3)
    ax3.axvline(x=avg_sev, color='black', linestyle='--')
    ax3.text(avg_sev + 0.5, len(stats) - 1, f'Середнє: {avg_sev:.1f}%')
    st.pyplot(fig3)

    st.write("#### Табличні дані")
    st.dataframe(stats)

# ==============================================================================
# СТОРІНКА 3: ПРОГНОЗ (Твій третій файл)
# ==============================================================================
elif page == "📈 Прогноз (Prophet)":
    st.title("📈 Прогноз кількості ДТП")
    st.markdown("Часові ряди та прогнозування за допомогою бібліотеки **Prophet**.")

    df_prophet = load_prophet_data()

    # Підготовка
    if 'accidentDate' in df_prophet.columns:
        df_prophet['Date'] = pd.to_datetime(df_prophet['accidentDate'])
    else:
        st.error("У файлі CSV немає колонки 'accidentDate'")
        st.stop()

    daily_df = df_prophet.groupby('Date').size().reset_index(name='y')
    daily_df.columns = ['ds', 'y']

    st.write(f"Завантажено даних за {len(daily_df)} днів.")

    periods = st.slider("На скільки днів прогнозувати вперед?", 7, 365, 30)

    if st.button("🔮 Сгенерувати прогноз"):
        with st.spinner('Тренуємо Prophet... Це може зайняти хвилину.'):
            # Модель на всіх даних
            m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
            m.add_country_holidays(country_name='UA')
            m.fit(daily_df)

            future = m.make_future_dataframe(periods=periods)
            forecast = m.predict(future)

        st.success("Прогноз готовий!")

        # Графік 1
        st.subheader("Тренд та прогноз")
        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        # Графік 2
        st.subheader("Компоненти (Сезонність)")
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)

        # Метрики на тестовому періоді (останні 30 днів з історії)
        st.markdown("---")
        st.write("### Оцінка точності (Backtesting)")
        test_days = 30
        train_df = daily_df.iloc[:-test_days]
        test_df = daily_df.iloc[-test_days:]

        m_test = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        m_test.add_country_holidays(country_name='UA')
        m_test.fit(train_df)
        forecast_test = m_test.predict(test_df)

        y_true = test_df['y'].values
        y_pred = forecast_test['yhat'].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE (Помилка)", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("Середнє ДТП/день", f"{daily_df['y'].mean():.2f}")