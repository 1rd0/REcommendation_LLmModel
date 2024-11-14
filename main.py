import pandas as pd
import requests
import faiss
import numpy as np

# Загрузка данных из CSV
df = pd.read_csv("/Users/kirillrabdel/OllamaRecMovie/netflix_titles.csv")

# Функция для создания текстового представления
def create_textual_representation(row):
    return f"""Type: {row['type']},
Title: {row['title']},
Director: {row['director']},
Cast: {row['cast']},
Released: {row['release_year']},
Genres: {row['listed_in']},

Description: {row['description']}"""

df['textual_representation'] = df.apply(create_textual_representation, axis=1)

# Настройка FAISS
dim = 1024
index = faiss.IndexFlatL2(dim)
X = np.zeros((len(df['textual_representation']), dim), dtype='float32')

# Функция для получения эмбеддингов через HTTP API с таймаутом
def get_embedding_from_ollama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                'model': 'mxbai-embed-large',
                'prompt': prompt
            },
            timeout=5  # Устанавливаем таймаут в 5 секунд
        )
        response.raise_for_status()
        data = response.json()

        if 'embedding' in data:
            return data['embedding']
        else:
            print("Ошибка: 'embedding' не найден в ответе:", data)
            return None

    except requests.exceptions.Timeout:
        print("Ошибка: Превышено время ожидания ответа от сервера.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к серверу Ollama: {e}")
        return None

# Заполняем матрицу эмбеддингами
#for i, prompt in enumerate(df['textual_representation']):
    if i % 30 == 0:
        print('Processed', str(i), 'instances')

    embedding = get_embedding_from_ollama(prompt)
    
    if embedding:
        X[i] = np.array(embedding, dtype='float32')
    else:
        print(f"Не удалось получить embedding для строки {i}")

# Добавляем вектора в FAISS индекс и сохраняем его
index.add(X)
index = faiss.read_index('/Users/kirillrabdel/OllamaRecMovie/index.faiss')

favorit_movie = df.iloc[1358]

response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                'model': 'mxbai-embed-large',
                'prompt': favorit_movie['textual_representation'] 
            },
        )
# Проверка на успешность запроса и получение эмбеддинга
data = response.json()
if 'embedding' in data:
    embedding = np.array(data['embedding'], dtype='float32')
    print("Эмбеддинг для выбранного фильма:", embedding)

    # Преобразуем одномерный массив в двумерный (1, dim)
    embedding = embedding.reshape(1, -1)

    # Выполняем поиск с помощью FAISS
    D, I = index.search(embedding, 5)
    print("Найденные ближайшие соседи:", I)
    print("Расстояния до соседей:", D)
else:
    print("Ошибка: 'embedding' не найден в ответе:", data)

best_matches= np.array(df['textual_representation'])[I.flatten()]
 

for match in best_matches:
    print("next movie")
    print(match)
    print()