# Chunking Evaluation 
---
❗❗❗ Данный репозиторий - это форк репозитория [brandonstarxel/chunking_evaluation](https://github.com/brandonstarxel/chunking_evaluation), адаптированный для работы с LLM GigaChat и GigaChatEmbeddings.
В данном проекте все реализованные в оригинальном репозитории функции и методы работают с использованием модели ```GigaChat-2-Max``` и ```EmbeddingsGigaR```

---

Этот пакет, разработанный в рамках исследования, подробно описанного в [Chroma Technical Report](https://research.trychroma.com/evaluating-chunking), предоставляет инструменты для разделения текста на фрагменты (чанкинг) и их оценки. Он позволяет пользователям сравнивать различные методы чанкинга и включает реализации нескольких новых стратегий.

## Особенности

- **Сравнение методов чанкинга**: Оценка и сравнение различных популярных стратегий чанкинга.
- **Новые методы чанкинга**: Реализации новых методов, таких как `ClusterSemanticChunker` и `LLMChunker`.
- **Фреймворк для оценки**: Инструменты для создания специализированных наборов данных и оценки качества извлечения информации в контексте AI-приложений.

## Установка

Вы можете установить пакет напрямую из GitHub:

```bash
pip install git+https://github.com/Asya02/chunking_evaluation.git
```

# Настройка использования модели EmbeddingsGigaR
```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class GigaEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        load_dotenv(find_dotenv())

        giga_embeds = GigaChatEmbeddings(
            verify_ssl_certs=False,
            model="EmbeddingsGigaR"
        )
        embeddings = giga_embeds.embed_documents(input)
        return embeddings

# Instantiate instance of ef
default_ef = GigaEmbeddingFunction()
```

# Оценка собственного кастомного чанкера
Этот пример показывает, как реализовать собственную логику чанкинга и оценить её производительность.
```python
from chunking_evaluation import BaseChunker, GeneralEvaluation
from chromadb.utils import embedding_functions

# Define a custom chunking class
class CustomChunker(BaseChunker):
    def split_text(self, text):
        # Custom chunking logic
        return [text[i:i+1200] for i in range(0, len(text), 1200)]

# Instantiate the custom chunker and evaluation
chunker = CustomChunker()
evaluation = GeneralEvaluation()

# Evaluate the chunker
results = evaluation.run(chunker, default_ef)

print(results)
# {'iou_mean': 0.17715979570301696, 'iou_std': 0.10619791407460026, 
# 'recall_mean': 0.8091207841640163, 'recall_std': 0.3792297991952294}
```

# Оценка кастомной функции эмбеддинга
```python
from chromadb import Documents, EmbeddingFunction, Embeddings

class MyEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        # embed the documents somehow
        return embeddings

# Instantiate instance of ef
default_ef = MyEmbeddingFunction()

# Evaluate the embedding function with a chunker
results = evaluation.run(chunker, default_ef)
```

# Использование и оценка ClusterSemanticChunker
Этот пример демонстрирует, как использовать наш ClusterSemanticChunker и как вы можете оценить его самостоятельно.
```python
    from chunking_evaluation import BaseChunker, GeneralEvaluation
    from chunking_evaluation.chunking import ClusterSemanticChunker
    from chromadb.utils import embedding_functions

    # Instantiate evaluation
    evaluation = GeneralEvaluation()

    # Instantiate chunker and run the evaluation
    chunker = ClusterSemanticChunker(default_ef, max_chunk_size=400)
    results = evaluation.run(chunker, default_ef)

    print(results)
    # {'iou_mean': 0.18255175232840098, 'iou_std': 0.12773219595465307, 
    # 'recall_mean': 0.8973469551927365, 'recall_std': 0.29042203879923994}
```

## Пайплайн для создания синтетического набора данных для предметной оценки

Вот шаги, которые вы можете выполнить для создания синтетического набора данных на основе ваших собственных корпусов для предметной оценки.

1. **Инициализация окружения**:

    ```python
    from chunking_evaluation import SyntheticEvaluation

    # Specify the corpora paths and output CSV file
    corpora_paths = [
        # 'path/to/txt',
    ]
    queries_csv_path = 'generated_queries_excerpts.csv'

    # Initialize the evaluation
    evaluation = SyntheticEvaluation(corpora_paths, queries_csv_path)
    ```

2. **Генерация запросов и отрывков**:

    ```python
    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts()
    ```

3. **Применение фильтров**:

    ```python
    # Apply filter to remove queries with poor excerpts
    evaluation.filter_poor_excerpts(threshold=0.36)
    
    # Apply filter to remove duplicates
    evaluation.filter_duplicates(threshold=0.6)
    ```

4. **Запуск оценки**:

    ```python
    from chunking_evaluation import BaseChunker

    # Define a custom chunking class
    class CustomChunker(BaseChunker):
        def split_text(self, text):
            # Custom chunking logic
            return [text[i:i+1200] for i in range(0, len(text), 1200)]

    # Instantiate the custom chunker
    chunker = CustomChunker()

    # Run the evaluation on the filtered data
    results = evaluation.run(chunker)
    print("Evaluation Results:", results)
    ```

2. **Опционально: Если генерация не может создать запросы, попробуйте approximate_excerpts**

    ```python
    # Generate queries and excerpts, and save to CSV
    evaluation.generate_queries_and_excerpts(approximate_excerpts=True)
    ```
