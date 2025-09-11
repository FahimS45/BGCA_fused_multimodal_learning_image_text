"""Utility functions."""

import torch
import numpy as np
import random
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List, Tuple, Dict, Set

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def filter_discriminative_words(texts: List[str], labels: List[int], top_n: int = 7) -> Tuple[List[str], Dict[int, Set[str]]]:
    """Filter discriminative words from texts."""
    custom_stopwords = ENGLISH_STOP_WORDS.difference({'no'})
    
    # Tokenize by class
    class_tokens = {0: [], 1: []}
    for text, label in zip(texts, labels):
        words = [word.lower().strip('.,():;-"\n') for word in text.split() 
                if word not in custom_stopwords]
        class_tokens[label].extend(words)
    
    # Find discriminative words
    freq_0 = Counter(class_tokens[0])
    freq_1 = Counter(class_tokens[1])
    removed_words = {0: set(), 1: set()}
    
    for target_class in [0, 1]:
        target_freq = freq_0 if target_class == 0 else freq_1
        other_freq = freq_1 if target_class == 0 else freq_0
        
        scores = {word: freq / (1 + other_freq.get(word, 0)) 
                 for word, freq in target_freq.items() if freq > 1}
        
        top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        removed_words[target_class] = {word for word, _ in top_words}
    
    all_removed = removed_words[0].union(removed_words[1])
    
    # Filter texts
    filtered_texts = []
    for text in texts:
        words = [word for word in text.split() 
                if word.lower().strip('.,():;-"\n') not in all_removed]
        filtered_texts.append(' '.join(words))
    
    return filtered_texts, removed_words