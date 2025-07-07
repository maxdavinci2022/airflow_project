import os
import dill
import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # <--- НОВЫЙ ИМПОРТ
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def pipeline():
    logging.info("Starting pipeline execution.")
    project_path = os.environ.get('PROJECT_PATH')
    if not project_path:
        logging.error(
            "PROJECT_PATH environment variable is not set. This should be set by Airflow or locally for testing.")
        raise ValueError("PROJECT_PATH is not set.")

    data_dir = os.path.join(project_path, 'data')
    model_dir = os.path.join(data_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)  # Убедимся, что папка для модели существует

    # Загрузка тренировочных данных
    train_data_path = os.path.join(data_dir, 'train')

    if not os.path.exists(train_data_path):
        logging.error(f"Training data directory not found at {train_data_path}")
        raise FileNotFoundError(f"Training data directory not found: {train_data_path}")

    train_files = [f for f in os.listdir(train_data_path) if f.endswith('.csv')]
    if not train_files:
        logging.error(f"No CSV training data files found in {train_data_path}")
        raise FileNotFoundError("No training data found.")

    all_train_data = []
    for f_name in train_files:
        f_path = os.path.join(train_data_path, f_name)
        all_train_data.append(pd.read_csv(f_path))

    df_train = pd.concat(all_train_data, ignore_index=True)
    logging.info(f"Loaded {len(df_train)} rows of training data.")

    # Определение признаков и целевой переменной (эти имена должны быть корректны!)
    target_column = 'price_category'

    numeric_features = ['year', 'price', 'odometer', 'lat', 'long']
    categorical_features = ['region', 'manufacturer', 'model', 'fuel', 'title_status', 'transmission', 'state']

    # Формируем полный список признаков для X
    features = numeric_features + categorical_features

    # Проверка, что колонки существуют в данных
    missing_features_in_data = [f for f in features if f not in df_train.columns]
    if missing_features_in_data:
        logging.error(f"Missing features in training data: {missing_features_in_data}")
        raise ValueError("Some specified features are not in the training data.")

    if target_column not in df_train.columns:
        logging.error(f"Target column '{target_column}' not found in training data.")
        raise ValueError(f"Target column '{target_column}' not found.")

    X = df_train[features]
    y = df_train[target_column]

    # Создание препроцессора для разных типов колонок
    # === ИЗМЕНЕННЫЙ БЛОК: ДОБАВЛЕНИЕ SimpleImputer ===
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Заполняем пропуски средним значением
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Заполняем пропуски наиболее частым значением
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    # === КОНЕЦ ИЗМЕНЕННОГО БЛОКА ===

    # Создание полного пайплайна
    pipeline_obj = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])

    # Параметры для GridSearchCV
    param_grid = {
        'classifier__C': [0.1, 1.0, 10.0],
        'classifier__penalty': ['l1', 'l2']
    }

    logging.info("Starting GridSearchCV for model training...")
    grid_search = GridSearchCV(pipeline_obj, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X, y)  # Теперь X должен быть без NaN после препроцессора

    best_pipeline = grid_search.best_estimator_
    logging.info(f"Best pipeline score: {grid_search.best_score_}")
    logging.info(f"Best parameters: {grid_search.best_params_}")

    # Сохраняем обученный пайплайн в файл с конкретным именем
    model_filename = 'model.pkl'
    model_filepath = os.path.join(model_dir, model_filename)

    try:
        with open(model_filepath, 'wb') as f:
            dill.dump(best_pipeline, f)
        logging.info(f"Pipeline saved successfully to {model_filepath}")
    except Exception as e:
        logging.error(f"Failed to save pipeline to {model_filepath}: {e}")
        raise

    logging.info("Pipeline execution finished.")


if __name__ == '__main__':
    os.environ['PROJECT_PATH'] = 'D:/Projects/_Skillbox/ds-intro/33_airflow/airflow_hw'
    logging.info("Running pipeline() locally...")
    pipeline()
    logging.info("Local pipeline() finished.")