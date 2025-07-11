import os
import re
from enum import Enum
from typing import List

import tiktoken
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions
from dotenv import find_dotenv, load_dotenv
from fuzzywuzzy import fuzz, process
from langchain_gigachat.chat_models.gigachat import GigaChat
from langchain_gigachat.embeddings import GigaChatEmbeddings

load_dotenv(find_dotenv())


class CustomGigaChatEmbeddings(GigaChatEmbeddings):
    def embed_documents(self, *args, **kwargs) -> List[List[float]]:
        """Embed documents using a GigaChat embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        for attempt in range(20):
            try:
                result = super().embed_documents(*args, **kwargs)
            except Exception as e:
                if attempt >= 19:
                    raise e
                else:
                    print(f"!!! Error: {e}, retrying...")
                    continue
        return result


class CustomGigaChat(GigaChat):
    def invoke(self, *args, **kwargs):
        for attempt in range(20):
            try:
                result = super().invoke(*args, **kwargs)
                finish_reason = result.response_metadata['finish_reason']
                if finish_reason != 'length':
                    return result
                else:
                    print("!!! Length exceeded, retrying...")
                    continue
            except Exception as e:
                if attempt >= 19:
                    raise e
                else:
                    print(f"!!! Error: {e}, retrying...")
                    continue
        return result


def find_query_despite_whitespace(document, query):

    # Normalize spaces and newlines in the query
    normalized_query = re.sub(r'\s+', ' ', query).strip()

    # Create a regex pattern from the normalized query to match any whitespace characters between words
    pattern = r'\s*'.join(re.escape(word) for word in normalized_query.split())

    # Compile the regex to ignore case and search for it in the document
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.search(document)

    if match:
        return document[match.start(): match.end()], match.start(), match.end()
    else:
        return None


def rigorous_document_search(document: str, target: str):
    """
    This function performs a rigorous search of a target string within a document. 
    It handles issues related to whitespace, changes in grammar, and other minor text alterations.
    The function first checks for an exact match of the target in the document. 
    If no exact match is found, it performs a raw search that accounts for variations in whitespace.
    If the raw search also fails, it splits the document into sentences and uses fuzzy matching 
    to find the sentence that best matches the target.

    Args:
        document (str): The document in which to search for the target.
        target (str): The string to search for within the document.

    Returns:
        tuple: A tuple containing the best match found in the document, its start index, and its end index.
        If no match is found, returns None.
    """
    if target.endswith('.'):
        target = target[:-1]

    if target in document:
        start_index = document.find(target)
        end_index = start_index + len(target)
        return target, start_index, end_index
    else:
        raw_search = find_query_despite_whitespace(document, target)
        if raw_search is not None:
            return raw_search

    # Split the text into sentences
    sentences = re.split(r'[.!?]\s*|\n', document)

    # Find the sentence that matches the query best
    best_match = process.extractOne(target, sentences, scorer=fuzz.token_sort_ratio)

    if best_match[1] < 98:
        return None

    reference = best_match[0]

    start_index = document.find(reference)
    end_index = start_index + len(reference)

    return reference, start_index, end_index


def get_openai_embedding_function():
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None:
        raise ValueError("You need to set an embedding function or set an OPENAI_API_KEY environment variable.")
    embedding_function = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv('OPENAI_API_KEY'),
        model_name="text-embedding-3-large"
    )
    return embedding_function


class GigaChatEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        giga_embeds = CustomGigaChatEmbeddings(
            verify_ssl_certs=False,
            model="EmbeddingsGigaR"
        )
        embeddings = giga_embeds.embed_documents(input)
        return embeddings


def get_gigachat_embedding_function():
    embedding_function = GigaChatEmbeddingFunction()
    return embedding_function


# Count the number of tokens in each page_content
def openai_token_count(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string, disallowed_special=()))
    return num_tokens


class Language(str, Enum):
    """Enum of the programming languages."""

    CPP = "cpp"
    GO = "go"
    JAVA = "java"
    KOTLIN = "kotlin"
    JS = "js"
    TS = "ts"
    PHP = "php"
    PROTO = "proto"
    PYTHON = "python"
    RST = "rst"
    RUBY = "ruby"
    RUST = "rust"
    SCALA = "scala"
    SWIFT = "swift"
    MARKDOWN = "markdown"
    LATEX = "latex"
    HTML = "html"
    SOL = "sol"
    CSHARP = "csharp"
    COBOL = "cobol"
    C = "c"
    LUA = "lua"
    PERL = "perl"