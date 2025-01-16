import array
from .file_handler import extract_text_from_file
import re
import xxhash
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Set, List, Tuple
import multiprocessing

# Constantes optimisées
CHUNK_SIZE = 5
MIN_WORD_LENGTH = 3
CLEAN_TEXT_REGEX = re.compile(r'[^\w\s]')
SPLIT_WORDS_REGEX = re.compile(r'\s+')
MAX_WORKERS = min(4, (multiprocessing.cpu_count() or 2))
BATCH_SIZE = 5000  # Taille optimale des batchs pour le traitement

@dataclass
class ProcessedText:
    """Classe pour stocker le texte prétraité et ses caractéristiques"""
    raw_text: str
    cleaned_text: str
    words: List[str]
    word_count: int
    fingerprints: Set[int]

# Cache global pour les textes traités
text_cache: Dict[str, ProcessedText] = {}

@lru_cache(maxsize=50000)
def clean_word(word: str) -> str:
    """Version optimisée du nettoyage de mots"""
    return word.lower() if len(word) >= MIN_WORD_LENGTH else ''

def get_fingerprints(text: str, k: int) -> Set[int]:
    """Calcul optimisé des empreintes"""
    if len(text) < k:
        return set()
    fingerprints = set()
    last_pos = len(text) - k + 1
    for i in range(0, last_pos, 2):  # Optimisation : pas de 2 pour réduire le nombre d'empreintes
        chunk = text[i:i + k]
        fingerprints.add(xxhash.xxh64(chunk).intdigest())
    return fingerprints

def preprocess_text(text: str, lang: str = 'french') -> ProcessedText:
    """Prétraitement optimisé avec mise en cache"""
    if not text:
        return ProcessedText("", "", [], 0, set())

    # Vérification du cache avec un hash rapide du texte brut
    text_hash = xxhash.xxh64(text).hexdigest()
    if text_hash in text_cache:
        return text_cache[text_hash]

    # Nettoyage et tokenization optimisés
    cleaned_text = CLEAN_TEXT_REGEX.sub(' ', text.lower())
    words = [w for w in SPLIT_WORDS_REGEX.split(cleaned_text) if len(w) >= MIN_WORD_LENGTH]
    
    # Création de l'objet ProcessedText
    processed = ProcessedText(
        raw_text=text,
        cleaned_text=' '.join(words),
        words=words,
        word_count=len(words),
        fingerprints=get_fingerprints(' '.join(words), CHUNK_SIZE)
    )
    
    # Mise en cache
    text_cache[text_hash] = processed
    return processed

def rabin_karp_similarity(text1: str, text2: str, k: int = None) -> float:
    """Rabin-Karp optimisé"""
    proc1 = preprocess_text(text1)
    proc2 = preprocess_text(text2)
    
    if proc1.cleaned_text == proc2.cleaned_text:
        return 100.0
    
    if not proc1.word_count or not proc2.word_count:
        return 0.0
    
    intersection = len(proc1.fingerprints & proc2.fingerprints)
    union = len(proc1.fingerprints | proc2.fingerprints)
    
    return (intersection / union * 100) if union > 0 else 0.0

def levenshtein_distance(text1: str, text2: str) -> float:
    """Levenshtein optimisé basé sur les mots"""
    proc1 = preprocess_text(text1)
    proc2 = preprocess_text(text2)
    
    if proc1.cleaned_text == proc2.cleaned_text:
        return 100.0
    
    if not proc1.word_count or not proc2.word_count:
        return 0.0
    
    # Utilisation de Counter pour optimiser
    words1 = Counter(proc1.words)
    words2 = Counter(proc2.words)
    
    common_words = sum((words1 & words2).values())
    total_words = sum((words1 | words2).values())
    
    return (common_words / total_words * 100) if total_words > 0 else 0.0

def lcs(text1: str, text2: str) -> float:
    """LCS optimisé"""
    proc1 = preprocess_text(text1)
    proc2 = preprocess_text(text2)
    
    if proc1.cleaned_text == proc2.cleaned_text:
        return proc1.word_count
    
    if not proc1.word_count or not proc2.word_count:
        return 0
    
    # Optimisation pour les textes très différents
    if not bool(set(proc1.words) & set(proc2.words)):
        return 0
    
    m, n = proc1.word_count, proc2.word_count
    if m < n:
        return lcs(text2, text1)
    
    current = array.array('L', [0] * (n + 1))
    for i, word1 in enumerate(proc1.words, 1):
        previous = current[:]
        for j, word2 in enumerate(proc2.words, 1):
            if word1 == word2:
                current[j] = previous[j-1] + 1
            else:
                current[j] = max(current[j-1], previous[j])
    
    return current[n]

def process_batch(batch: List[Tuple[int, int, str, str]]) -> List[Tuple[int, int, float]]:
    """Traitement optimisé d'un batch de comparaisons"""
    results = []
    for i, j, text1, text2 in batch:
        # Calcul parallèle des trois similarités
        with ThreadPoolExecutor(max_workers=3) as executor:
            rk_future = executor.submit(rabin_karp_similarity, text1, text2)
            lev_future = executor.submit(levenshtein_distance, text1, text2)
            lcs_future = executor.submit(
                lambda: (lcs(text1, text2) / max(len(text1.split()), len(text2.split()))) * 100 
                if text1 and text2 else 0
            )
            
            similarity = round(
                (rk_future.result() * 0.3 + 
                 lev_future.result() * 0.3 + 
                 lcs_future.result() * 0.4), 2
            )
            
        results.append((i, j, similarity))
    return results

def process_files(files: List) -> Dict:
    """Traitement optimisé de multiples fichiers"""
    try:
        n = len(files)
        if n < 2:
            raise ValueError("Au moins deux fichiers sont nécessaires")

        # Prétraitement parallèle des fichiers
        similarity_matrix = np.zeros((n, n), dtype=np.float32)
        filenames = [f.filename for f in files]
        
        # Extraction parallèle du texte
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            text_futures = {executor.submit(extract_text_from_file, f): i 
                          for i, f in enumerate(files)}
            
            texts = [""] * n
            for future in as_completed(text_futures):
                idx = text_futures[future]
                texts[idx] = future.result()
        
        # Pré-remplissage pour fichiers identiques
        for i in range(n):
            similarity_matrix[i, i] = 100.0
            for j in range(i + 1, n):
                if filenames[i] == filenames[j] or texts[i] == texts[j]:
                    similarity_matrix[i, j] = similarity_matrix[j, i] = 100.0
        
        # Création des batchs de comparaisons
        batches = []
        current_batch = []
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] == 0:
                    current_batch.append((i, j, texts[i], texts[j]))
                    if len(current_batch) >= BATCH_SIZE:
                        batches.append(current_batch)
                        current_batch = []
        if current_batch:
            batches.append(current_batch)
        
        # Traitement parallèle des batchs
        total_batches = len(batches)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_batch, batch) for batch in batches]
            
            for batch_idx, future in enumerate(as_completed(futures), 1):
                for i, j, similarity in future.result():
                    similarity_matrix[i, j] = similarity_matrix[j, i] = similarity
                
                print(f"Progression: batch {batch_idx}/{total_batches}")
        
        # Nettoyage du cache
        text_cache.clear()
        
        return {
            'matrix': similarity_matrix.tolist(),
            'filenames': filenames
        }
        
    except Exception as e:
        text_cache.clear()  # Nettoyage en cas d'erreur
        raise Exception(f"Erreur lors du traitement des fichiers: {str(e)}")

def get_similarity_message(percentage: float) -> str:
    """Fonction de message optimisée avec des seuils prédéfinis"""
    if percentage >= 90: return "Plagiat très probable ! Les textes sont presque identiques."
    if percentage >= 70: return "Fort niveau de similarité. Il y a de fortes chances de plagiat."
    if percentage >= 50: return "Similarité modérée. Certains passages pourraient être copiés."
    if percentage >= 30: return "Faible similarité. Quelques ressemblances détectées."
    return "Très peu de similarité. Probablement pas de plagiat."

def compare_two_files(file1, file2) -> Dict:
    """Comparaison optimisée de deux fichiers"""
    try:
        if file1.filename == file2.filename:
            return {
                'similarity': 100.0,
                'message': "Les fichiers sont identiques.",
                'details': {'rabin_karp': 100.0, 'levenshtein': 100.0, 'lcs': 100.0}
            }
        
        # Extraction parallèle du texte
        with ThreadPoolExecutor(max_workers=2) as executor:
            text1_future = executor.submit(extract_text_from_file, file1)
            text2_future = executor.submit(extract_text_from_file, file2)
            text1 = text1_future.result()
            text2 = text2_future.result()
        
        if text1 == text2:
            return {
                'similarity': 100.0,
                'message': "Les contenus des fichiers sont identiques.",
                'details': {'rabin_karp': 100.0, 'levenshtein': 100.0, 'lcs': 100.0}
            }
        
        # Calcul parallèle des similarités
        with ThreadPoolExecutor(max_workers=3) as executor:
            rk_future = executor.submit(rabin_karp_similarity, text1, text2)
            lev_future = executor.submit(levenshtein_distance, text1, text2)
            lcs_future = executor.submit(
                lambda: (lcs(text1, text2) / max(len(text1.split()), len(text2.split()))) * 100 
                if text1 and text2 else 0
            )
            
            rk_similarity = rk_future.result()
            lev_similarity = lev_future.result()
            lcs_similarity = lcs_future.result()
        
        final_similarity = round(
            (rk_similarity * 0.3 + lev_similarity * 0.3 + lcs_similarity * 0.4), 2
        )
        
        return {
            'similarity': final_similarity,
            'message': get_similarity_message(final_similarity),
            'details': {
                'rabin_karp': round(rk_similarity, 2),
                'levenshtein': round(lev_similarity, 2),
                'lcs': round(lcs_similarity, 2)
            }
        }
        
    except Exception as e:
        raise Exception(f"Erreur lors de la comparaison des fichiers: {str(e)}")