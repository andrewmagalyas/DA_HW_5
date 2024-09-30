"""
Напишіть програму на Python,
щоб завантажити дані ірису з вказаного файлу csv у dataframe та надрукувати форму даних, тип даних та перші 3 рядки.
за допомогою Scikit-learn, щоб надрукувати ключі, кількість рядків-стовпців, назви ознак та опис даних Ірису.
щоб переглянути базові статистичні деталі, як-от перцентиль, середнє, стандартне відхилення тощо даних ірису.
щоб отримати спостереження кожного виду (сетоза, версиколор, віргініка) з даних ірису.
щоб створити графік для отримання загальної статистики даних Ірис.
Напишіть програму на Python, щоб створити стовпчасту діаграму для визначення частоти трьох видів Ірис.
для розподілу набору даних ірисів на його атрибути (X) та мітки (y).
Змінна X містить перші чотири стовпці (тобто атрибути), а y містить мітки набору даних. Натисніть тут, щоб побачити приклад розв'язку.
за допомогою Scikit-learn для розділення набору даних ірисів на 70% тренувальних даних та 30% тестових даних.
З загальної кількості 150 записів, набір для тренування міститиме 120 записів, а тестовий набір - 30 з цих записів.
Виведіть обидва набори даних.
Напишіть програму на Python за допомогою Scikit-learn для перетворення стовпців видів у числовий стовпець набору даних ірисів.
Для кодування цих даних кожне значення перетворіть на число. Наприклад, Iris-setosa:0, Iris-versicolor:1 та Iris-virginica:2.
Тепер виведіть набір даних ірисів на 80% тренувальних даних і 20% тестових даних.
З загальної кількості 150 записів, набір для тренування міститиме 120 записів, а тестовий набір - 30 з цих записів.
Виведіть обидва набори даних.
Напишіть програму на Python за допомогою Scikit-learn для розділення набору даних ірисів на 70% тренувальних даних
та 30% тестових даних. З загальної кількості 150 записів, набір для тренування міститиме 105 записів,
а тестовий набір - 45 з цих записів. Прогнозуйте відповідь для тестового набору даних
(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm) за допомогою алгоритму найближчих сусідів (K Nearest Neighbor Algorithm).
Використовуйте 5 як кількість сусідів.
"""

import argparse
import logging
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_iris_data(file_path: str):
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.columns = [col.replace(' (cm)', '').replace(' ', '_') for col in df.columns]
    df.to_csv(file_path, index=False)
    return df


file_path = 'iris.csv'
df = load_iris_data(file_path)
print(df.head())


def load_and_describe_data(file_path: str) -> pd.DataFrame:
    """Завантажує дані з файлу та виводить їх опис."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Форма даних: {df.shape}")
        logging.info(f"\nТипи даних:\n{df.dtypes}")
        logging.info(f"\nПерші 3 рядки:\n{df.head(3)}")
        return df
    except FileNotFoundError:
        logging.error(f"Файл {file_path} не знайдено. Перевірте шлях до файлу.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Виникла помилка при завантаженні даних: {e}")
        return pd.DataFrame()


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Готує дані для навчання моделі: атрибути та мітки."""
    if 'target' in df.columns:
        X = df.iloc[:, :-1]
        y = df['target']
        le = LabelEncoder()
        y = le.fit_transform(y)
        return X, y
    else:
        logging.error("Колонка 'target' відсутня в наборі даних.")
        return pd.DataFrame(), np.array([])


def visualize_data(df: pd.DataFrame, save_path: str = None) -> None:
    """Візуалізує дані через pairplot та boxplot."""
    if 'target' not in df.columns:
        logging.warning("Колонка 'target' відсутня в наборі даних.")
        return

    # Pairplot
    sns.pairplot(df, hue="target")
    if save_path:
        plt.savefig(f"{save_path}_pairplot.png")
    else:
        plt.show()
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    species_count = df['target'].value_counts()
    species_count.plot(kind='bar', ax=ax1, color=['red', 'green', 'blue'])
    ax1.set_title('Частота видів Ірису')
    ax1.set_ylabel('Кількість')
    ax1.set_xlabel('Вид')

    df.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], by='target', ax=ax2)
    ax2.set_title('Розподіл характеристик за видами')

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_boxplot.png")
    else:
        plt.show()
    plt.close()


def split_and_train_model(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.3, n_neighbors: int = 5) -> Tuple:
    """Розділяє дані на тренувальні та тестові, тренує модель KNN та повертає точність."""
    if X.empty or y.size == 0:
        logging.error("Набір даних порожній або некоректний.")
        return None, None, None, None, None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info(f"Точність моделі: {accuracy:.2f}")
    logging.info("\nЗвіт класифікації:\n" + classification_report(y_test, y_pred))

    cv_scores = cross_val_score(knn, X, y, cv=5)
    logging.info(f"Середня точність крос-валідації: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

    return X_train, X_test, y_train, y_test, y_pred


def main(file_path: str, test_size: float, n_neighbors: int, save_path: str):
    df = load_and_describe_data(file_path)

    if not df.empty:
        visualize_data(df, save_path)

        X, y = prepare_data(df)
        if not X.empty and y.size > 0:
            X_train, X_test, y_train, y_test, y_pred = split_and_train_model(X, y, test_size, n_neighbors)
            if X_train is not None:
                logging.info(f"Тренувальні дані: {X_train.shape}, Тестові дані: {X_test.shape}")
                logging.info(f"Прогнозовані відповіді для тестового набору:\n{y_pred}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аналіз даних Iris")
    parser.add_argument("--file", type=str, default="iris.csv", help="Шлях до файлу з даними")
    parser.add_argument("--test_size", type=float, default=0.3, help="Розмір тестової вибірки")
    parser.add_argument("--n_neighbors", type=int, default=5, help="Кількість сусідів для KNN")
    parser.add_argument("--save_path", type=str, help="Шлях для збереження графіків")
    args = parser.parse_args()

    main(args.file, args.test_size, args.n_neighbors, args.save_path)
