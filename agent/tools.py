import json
import random
from typing import List, Dict
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
import logging

logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)

# reasoning=False: Qwen3 "thinks" by default, and its <think>...</think>
# block leaks into response.content, breaking json.loads() below.
_generation_llm = ChatOllama(model="qwen3:8b", temperature=0.8, reasoning=False)
_translation_llm = ChatOllama(model="qwen3:8b", temperature=0.0, reasoning=False)

# High-quality fallback vocabulary list in case the local LLM is temporarily busy or offline
FALLBACK_VOCABULARY = {
    "english": {
        "beginner": ["apple", "house", "book", "friend", "school", "family", "water", "bread", "happy", "morning"],
        "intermediate": ["challenge", "develop", "decision", "community", "schedule", "dangerous", "standard",
                         "creative"],
        "advanced": ["meticulous", "ephemeral", "cacophony", "exacerbate", "scrutinize", "ambiguity", "transient"]
    },
    "spanish": {
        "beginner": ["manzana", "casa", "libro", "amigo", "escuela", "familia", "agua", "pan", "feliz", "mañana"],
        "intermediate": ["desafío", "desarrollo", "decisión", "comunidad", "horario", "peligroso", "estándar",
                         "creativo"],
        "advanced": ["meticuloso", "efímero", "cacofonía", "exacerbar", "escudriñar", "ambigüedad", "transitorio"]
    }
}


@tool
def get_n_random_words(language: str, n: int) -> List[str]:
    """
    Dynamically generates a specified number of random, real-world, conversational words in a target language.

    Args:
        language: The target language, e.g. 'english', 'french', 'german'.
    """
    lang = language.lower().strip()

    prompt = f"""
    Generate a list of exactly {n} unique, highly common, everyday conversational words in {lang}.
    Do not include obscure words, abbreviations, names, or technical jargon.
    Return ONLY a valid JSON array of strings. No conversational text, no markdown formatting.
    Example output format: ["word1", "word2", "word3"]
    """

    try:
        response = _generation_llm.invoke(prompt)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        words = json.loads(clean_content)
        if isinstance(words, list) and len(words) > 0:
            return [str(w).lower().strip() for w in words[:n]]
    except Exception as e:
        print(f"generation failed: {e}")
        pass

    # Fallback logic if LLM generation fails
    all_fallbacks = []
    for level in FALLBACK_VOCABULARY.get(lang, {}):
        all_fallbacks.extend(FALLBACK_VOCABULARY[lang][level])
    return random.sample(all_fallbacks, min(n, len(all_fallbacks)))


@tool
def get_n_random_words_by_difficulty_level(language: str, difficulty_level: str, n: int) -> List[str]:
    """
    Dynamically generates real, meaningful words filtered by a given difficulty level ('beginner', 'intermediate', 'advanced').

    Args:
        language: The target language, e.g. 'english', 'french', 'german'.
        difficulty_level: The vocabulary tier ('beginner', 'intermediate', or 'advanced').
        n: The number of words to retrieve.
    """
    lang = language.lower().strip()

    diff = difficulty_level.lower().strip()
    if diff not in ["beginner", "intermediate", "advanced"]:
        diff = "beginner"

    prompt = f"""
    Generate a list of exactly {n} unique, everyday conversational words in {lang} suitable for a {diff} level language learner.
    Do not include obscure words, names, abbreviations, or symbols.
    Return ONLY a valid JSON array of strings. No other text.
    Example output format: ["word1", "word2", "word3"]
    """

    try:
        response = _generation_llm.invoke(prompt)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        words = json.loads(clean_content)
        if isinstance(words, list) and len(words) > 0:
            return [str(w).lower().strip() for w in words[:n]]
    except Exception as e:
        print(f"generation failed: {e}")
        pass

    # Fallback logic if LLM generation fails
    pool = FALLBACK_VOCABULARY.get(lang, {}).get(diff, [])
    return random.sample(pool, min(n, len(pool)))


@tool
def translate_words(random_words: List[str], source_language: str, target_language: str) -> List[Dict[str, str]]:
    """
    Translates a list of words from a source language to a target language dynamically.

    Args:
        random_words: List of words to translate.
        source_language: The language of the words in the list.
        target_language: The language to translate the words into.
    """
    results = []
    if not random_words:
        return results

    prompt = f"""
    Translate the following list of {source_language} words into {target_language}.
    Words list: {random_words}

    Provide your output strictly as a valid JSON array of objects, where each object has "original" and "translation" keys.
    No conversational text, no introductions, no markdown formatting.
    Example output:
    [{{"original": "dog", "translation": "perro"}}]
    """

    # Rebuild the result in the same order/length as random_words, regardless
    # of how the model ordered its response. Any word the model dropped or
    # renamed falls back to the "[word translation]" placeholder below.
    try:
        response = _translation_llm.invoke(prompt)
        clean_content = response.content.replace("```json", "").replace("```", "").strip()
        translations = json.loads(clean_content)
        if isinstance(translations, list):
            lookup = {
                str(t.get("original", "")).lower().strip(): t.get("translation")
                for t in translations if isinstance(t, dict)
            }
            return [
                {"original": w, "translation": lookup.get(w.lower().strip(), f"[{w} translation]")}
                for w in random_words
            ]
    except Exception as e:
        print(f"generation failed: {e}")
        pass

    # Basic backup fallback if translation call fails
    for word in random_words:
        results.append({
            "original": word,
            "translation": f"[{word} translation]"
        })
    return results