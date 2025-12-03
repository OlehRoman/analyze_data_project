import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import folium
from sklearn.cluster import DBSCAN

st.set_page_config(page_title="Аналіз ДТП Львів", layout="wide")

st.title("🗺️ Інтерактивна карта ДТП у Львові")
st.markdown("Змінюйте параметри зліва, щоб оновити кластеризацію.")


@st.cache_data
def load_data():
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


df = load_data()

st.sidebar.header("🔍 Фільтри")

# Фільтр по роках (якщо у тебе є колонка Year)
# Якщо нема, цей блок можна закоментувати або додати extraction року
if 'Year' in df.columns:
    years = sorted(df['Year'].unique())
    selected_years = st.sidebar.slider("Оберіть роки", min_value=min(years), max_value=max(years),
                                       value=(min(years), max(years)))
    df_filtered = df[(df['Year'] >= selected_years[0]) & (df['Year'] <= selected_years[1])]
else:
    df_filtered = df.copy()

# Фільтр по годинах
hour_range = st.sidebar.slider("Час доби", 0, 23, (0, 23))
df_filtered = df_filtered[(df_filtered['Hour'] >= hour_range[0]) & (df_filtered['Hour'] <= hour_range[1])]

# Вибір причин
all_causes = sorted(df['Simple_Cause'].unique())
selected_causes = st.sidebar.multiselect("Причини ДТП", all_causes, default=all_causes)
df_filtered = df_filtered[df_filtered['Simple_Cause'].isin(selected_causes)]

# Налаштування DBSCAN "на льоту"
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Налаштування алгоритму")
eps_meters = st.sidebar.slider("Радіус кластера (метри)", 20, 200, 70)
min_samples = st.sidebar.slider("Мін. к-сть аварій для кластера", 2, 20, 5)

# --- 4. ЛОГІКА КЛАСТЕРИЗАЦІЇ ---
# --- 4. ЛОГІКА КЛАСТЕРИЗАЦІЇ ТА МЕТРИКИ ---
if len(df_filtered) > 0:
    # 1. Підготовка та навчання
    coords = df_filtered[['latitude', 'longitude']].values
    coords_rad = np.radians(coords)

    kms_per_radian = 6371.0
    epsilon = (eps_meters / 1000) / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine', algorithm='ball_tree').fit(coords_rad)
    df_filtered['Cluster'] = db.labels_

    # 2. Розрахунок статистики шуму
    total_accidents = len(df_filtered)
    noise_count = np.sum(df_filtered['Cluster'] == -1)
    clustered_count = total_accidents - noise_count
    noise_percent = (noise_count / total_accidents) * 100
    clusters_found = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)

    # 3. ВІДОБРАЖЕННЯ МЕТРИК (У 3 колонки)
    st.markdown("### 📊 Загальна статистика")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric(
        label="Всього ДТП (у вибірці)",
        value=total_accidents
    )

    kpi2.metric(
        label="Знайдено зон концентрації",
        value=clusters_found,
        delta=f"{clustered_count} аварій"  # Показує дрібним шрифтом скільки аварій в кластерах
    )

    kpi3.metric(
        label="Поодинокі випадки (Шум)",
        value=f"{noise_percent:.1f}%",
        delta_color="off"  # Щоб колір був нейтральним
    )

    # Додаткова метрика: Середня к-сть ДТП на кластер
    avg_accidents = clustered_count / clusters_found if clusters_found > 0 else 0
    kpi4.metric(
        label="Середнє ДТП на зону",
        value=f"{avg_accidents:.1f}"
    )

    st.markdown("---")  # Розділювач

    # 4. Агрегація для карти
    cluster_stats = df_filtered[df_filtered['Cluster'] != -1].groupby('Cluster').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'Simple_Cause': lambda x: x.mode()[0] if not x.mode().empty else 'Інше',
        'Hour': lambda x: x.mode()[0] if not x.mode().empty else 0,
        'accidentDay': 'count'
    }).rename(columns={'accidentDay': 'AccidentCount'})

    # --- 5. ВІДОБРАЖЕННЯ КАРТИ (Тут код такий самий, як був) ---
    m = folium.Map(location=[49.8397, 24.0297], zoom_start=12)

    colors = {
        'Швидкість': 'red', 'Алкоголь': 'black', 'Перехрестя/Світлофор': 'orange',
        'Пішохід': 'purple', 'Маневрування': 'blue', 'Обгін/Зустрічка': 'darkred',
        'Дистанція': 'cadetblue', 'Невідомо': 'lightgray', 'Інше': 'green'
    }

    for cluster_id, row in cluster_stats.iterrows():
        cause = row['Simple_Cause']
        color = colors.get(cause, 'gray')

        radius = min(6 + (np.log1p(row['AccidentCount']) * 4), 25)

        popup_html = f"""
        <div style="font-family: Arial;">
            <b>Зона #{cluster_id}</b><br>
            <span style="color:{color}">{cause}</span><br>
            ДТП: {row['AccidentCount']}
        </div>
        """

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=radius,
            color=color,
            fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=f"{cause}: {row['AccidentCount']}"
        ).add_to(m)

    st_folium(m, width=1000, height=600)

    # Таблиця знизу
    st.markdown("### Деталі по зонах")
    st.dataframe(cluster_stats.sort_values(by='AccidentCount', ascending=False), use_container_width=True)

else:
    st.warning("Немає даних для відображення. Змініть фільтри.")