import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

# Укажите абсолютный путь к папке airflow_hw ВНУТРИ Docker-контейнера.
# Этот путь должен совпадать с правой частью монтирования, которое вы добавили в docker-compose.yaml.
# Это абсолютно правильно для Airflow.
path = '/opt/airflow/airflow_hw'

# Добавим путь к коду проекта в переменную окружения, чтобы он был доступен python-процессу
os.environ['PROJECT_PATH'] = path
# Добавим путь к коду проекта в $PATH, чтобы импортировать функции
sys.path.insert(0, path)

from modules.pipeline import pipeline
from modules.predict import predict # <--- ДОБАВЛЕНО: Импорт функции predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule="00 15 * * *",
        default_args=args,
        catchup=False # Обычно полезно добавить catchup=False для DAG, который не должен запускаться для прошлых дат
) as dag:
    pipeline_task = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline, # Вызывается функция pipeline напрямую
    )

    predict_task = PythonOperator( # <--- ДОБАВЛЕНО: Задача для предсказания
        task_id='predict',
        python_callable=predict, # Вызывается функция predict напрямую
    )

    pipeline_task >> predict_task # <--- ДОБАВЛЕНО: Зависимость: сначала pipeline, потом predict