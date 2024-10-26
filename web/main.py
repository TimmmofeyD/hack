import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from plotly_roc import metrics, graphs
import random



st.set_page_config(page_title="Прогнозирование раннего выхода на пенсию",
                           page_icon="⚰️", )
st.title("Аналитика пенсионных данных")

st.sidebar.title("Навигация")
section = st.sidebar.radio("Перейти к разделу",
                           ("Загрузка данных", "Отображение данных", "Анализ данных", "Результаты прогнозирования"))

# Раздел 1: Загрузка данных
if section == "Загрузка данных":
    st.header("Загрузка данных")

    # Загрузка файлов
    uploaded_file1 = st.file_uploader("Загрузите файл с данными клиентов", type=["csv", "xlsx"])
    uploaded_file2 = st.file_uploader("Загрузите файл с операциями по счетам", type=["csv", "xlsx"])

    if uploaded_file1 and uploaded_file2:
        # Чтение данных
        df_clients = pd.read_csv(uploaded_file1) if uploaded_file1.name.endswith('.csv') else pd.read_excel(
            uploaded_file1)
        df_operations = pd.read_csv(uploaded_file2) if uploaded_file2.name.endswith('.csv') else pd.read_excel(
            uploaded_file2)
        st.success("Файлы успешно загружены.")
        st.write("Пример данных клиентов:", df_clients.head())
        st.write("Пример данных операций:", df_operations.head())
    else:
        st.info("Пожалуйста, загрузите оба файла для продолжения.")

# Раздел 2: Отображение данных
elif section == "Отображение данных":
    st.header("Отображение данных")

    # Вывод таблиц с данными
    if 'df_clients' in locals() and 'df_operations' in locals():
        st.write("Данные клиентов:")
        #st.dataframe(df_1)

        st.write("Данные операций:")
        #st.dataframe(df_2)
    else:
        st.warning("Сначала загрузите данные в разделе 'Загрузка данных'.")

# Раздел 3: Анализ данных
elif section == "Анализ данных":
    st.header("Анализ данных")

    # Генерация и отображение roc-auc
    #if 'df_clients' in locals() and 'df_operations' in locals():
    random.seed(42)
    ns, ps = (80, 120)
    labels = [0] * ns + [1] * ps
    probas = [random.normalvariate(0.4, 0.25) for _ in range(ns)] + [random.normalvariate(0.7, 0.15) for _ in
                                                                         range(ps)]
    metrics_df = metrics.metrics_df(labels, probas)
    st.plotly_chart(graphs.roc_curve(metrics_df, line_name="Cat vs Dog", line_color="crimson", cm_labels=["CAT", "DOG"],
                                     fig_size=(600, 600)), use_container_width=True)
    st.plotly_chart(graphs.precision_recall_curve(metrics_df, line_name="Cat vs Dog", line_color="crimson", cm_labels=["CAT", "DOG"],
                                     fig_size=(600, 600)), use_container_width=True)


# Раздел 4: Результаты прогнозирования
elif section == "Результаты прогнозирования":
    st.header("Результаты прогнозирования")

    # Ввод параметров для прогноза
    st.write("Введите параметры для прогнозирования.")

    # Поля ввода для параметров (пример)
    age = st.number_input("Возраст клиента", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Пол клиента", options=["Мужчина", "Женщина"])
    account_duration = st.slider("Срок действия договора (в годах)", 0, 50, 5)

    # Кнопка для выполнения прогноза
    if st.button("Выполнить прогноз"):
        st.write("Прогнозируем...")

        # Пример вывода результатов (здесь будет подключена модель машинного обучения)
        # Placeholder для результатов прогноза
        st.write("Результат: Клиент имеет высокую вероятность раннего выхода на пенсию.")
        st.write("Рекомендуемый тип выплат: Ежемесячные выплаты.")
    else:
        st.info("Заполните параметры и нажмите 'Выполнить прогноз' для получения результатов.")
