import os
import pandas as pd
import dill
import logging
import json  # <--- НОВЫЙ ИМПОРТ: добавляем модуль json
from sklearn.pipeline import Pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def predict():
    project_path = os.environ.get('PROJECT_PATH')

    if not project_path:
        logging.error(
            "PROJECT_PATH environment variable is not set. This should be set by Airflow or locally for testing.")
        raise ValueError("PROJECT_PATH is not set.")

    data_dir = os.path.join(project_path, 'data')
    test_data_dir = os.path.join(data_dir, 'test')
    model_dir = os.path.join(data_dir, 'model')
    predictions_dir = os.path.join(data_dir, 'predictions')

    os.makedirs(predictions_dir, exist_ok=True)

    logging.info(f"Project path: {project_path}")
    logging.info(f"Test data directory: {test_data_dir}")
    logging.info(f"Model directory: {model_dir}")
    logging.info(f"Predictions directory: {predictions_dir}")

    # 1. Загрузка обученной модели
    model_filename = 'model.pkl'
    model_path = os.path.join(model_dir, model_filename)

    logging.info(f"Attempting to load model from: {model_path}")

    if not os.path.exists(model_path):
        logging.error(
            f"Error: Model file not found at {model_path}. Please ensure pipeline has been run successfully to create 'model.pkl'.")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with open(model_path, 'rb') as f:
            model = dill.load(f)
        logging.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        raise

    # 2. Сбор всех файлов тестовых данных
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.json')]
    if not test_files:
        logging.warning(f"No JSON files found in {test_data_dir}. No predictions will be made.")
        return

    all_predictions = []

    numeric_features = ['year', 'price', 'odometer', 'lat', 'long']
    categorical_features = ['region', 'manufacturer', 'model', 'fuel', 'title_status', 'transmission', 'state']
    features_for_prediction = numeric_features + categorical_features

    # 3. Делаем предсказания для каждого файла
    for file_name in test_files:
        file_path = os.path.join(test_data_dir, file_name)
        try:
            # === ИЗМЕНЕНИЕ ЗДЕСЬ: ЧТЕНИЕ JSON через модуль json ===
            with open(file_path, 'r', encoding='utf-8') as f:  # Указываем кодировку UTF-8
                json_data = json.load(f)
            # Если файл содержит один JSON-объект, преобразуем его в список из одного объекта
            # чтобы pandas создал DataFrame с одной строкой.
            df_test = pd.DataFrame([json_data])
            # === КОНЕЦ ИЗМЕНЕНИЙ ===

            test_ids = df_test['id'] if 'id' in df_test.columns else range(len(df_test))

            df_test_features = df_test[features_for_prediction]

            predictions = model.predict(df_test_features)

            predictions_df = pd.DataFrame({'id': test_ids, 'prediction': predictions})
            all_predictions.append(predictions_df)
            logging.info(f"Predictions made for {file_name}")
        except KeyError as ke:
            logging.error(
                f"Missing required feature in test file {file_name}: {ke}. Please ensure test data has all features used for training.")
            raise
        except json.JSONDecodeError as jde:  # Добавляем обработку ошибок парсинга JSON
            logging.error(f"Error decoding JSON from file {file_name}: {jde}. Please check file format and content.")
            raise
        except Exception as e:
            logging.error(f"Error processing file {file_name}: {e}")
            raise

            # 4. Объединяем предсказания
    if all_predictions:
        final_predictions_df = pd.concat(all_predictions, ignore_index=True)

        # 5. Сохраняем предсказания в csv-формате
        output_path = os.path.join(predictions_dir, 'predictions.csv')
        final_predictions_df.to_csv(output_path, index=False)
        logging.info(f"All predictions saved to {output_path}")
    else:
        logging.info("No predictions generated.")


if __name__ == '__main__':
    os.environ['PROJECT_PATH'] = 'D:/Projects/_Skillbox/ds-intro/33_airflow/airflow_hw'

    logging.info("Running predict() locally...")
    predict()
    logging.info("Local predict() finished.")