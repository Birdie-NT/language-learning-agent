import os
import json
import random

from langchain_core.tools  import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


translation_model = ChatOllama(
    model = "llama3.2:3b",
    temperature = 0.7
)

# The @tool decorator tells LangGraph that this function is an official capability
# that the AI Agent can choose to execute when it needs to look up words.
@tool
def get_n_random_words(language: str, n:int, ) -> list:
    """
        Looks up and returns a list of N random words from the cleaned dataset
        for a specific target language.
        """

    # 1. Construct the dynamic system file path based on the user's requested language.
    #    Example output path: 'data/spanish/word-list-cleaned.json'
    path = os.path.join("data", f"{language}", "word-list-cleaned.json")

    # 2. Open the specific JSON file and load the dictionary dataset into memory.
    with open(path) as f:
        word_list = json.load(f)

        # 3. Convert n to an integer to prevent crashes from state string inputs,
        #    and cap it to the total available words to prevent sample size errors.
        safe_n = min(int(n), len(word_list))

        random_keys = random.sample(list(word_list.keys()), safe_n)
        random_word_dict = {k: word_list[k] for k in random_keys}

    # 4. Extract just the raw text token (the actual string word) from each selected item object.
    random_words = [item["word"] for item in random_word_dict.values()]

    # 5. Send the final list of words back to the calling LangGraph agent state loop.
    return random_words


@tool

def get_n_random_words_by_difficulty_level(language: str,
                                           difficulty_level: str,
                                           n:int) -> list:
    """
    Retrieves a specified number of random words filtered by a given difficulty level
    from a word list corresponding to a specific language. The function reads the
    word list from a JSON file located in the directory `data/{language}/word-list-cleaned.json`.

    :param language: The language of the word list to be used.
    :type language: str
    :param difficulty_level: The difficulty level to filter words by. Possible values
        depend on the data structure in the JSON file.The only valid values are 'beginner',
        'intermediate', and 'advanced'.
    :type difficulty_level: str
    :param n: The number of random words to retrieve.
    :type n: int
    :return: A list containing `n` random words filtered by the specified difficulty level.
    :rtype: list
    """


    path = os.path.join("data", f"{language}" , "word-list-cleaned.json")

    with open(path) as f:
        word_list = json.load(f)

    # 1. Filter dictionary items matching the requested difficulty level string
    #    (e.g., 'beginner', 'intermediate', 'advanced')
    words_filtered_by_difficulty = {
        k: v for k, v in word_list.items()
        if v.get("word_difficulty") == difficulty_level.lower().strip()
    }

    # 2. Fallback: If no words match the filter, avoid crashing by using the whole list
    if not words_filtered_by_difficulty:
        words_filtered_by_difficulty = word_list

    # 3. SAFE GUARDS: Cast n to int to handle UI string states,
    #    and cap it to available keys to prevent sample size errors.
    safe_n = min(int(n), len(words_filtered_by_difficulty))

    # 4. Extract truly random samples out of the filtered sub-pool
   # random_keys = random.sample(list(words_filtered_by_difficulty.keys()), safe_n)
   # random_word_dict = {k: words_filtered_by_difficulty[k] for k in random_keys}

    sampled_data = random.sample(list(word_list.values()), safe_n)

    # 5. Extract the string value token to return clean vocabulary words
    random_words = [item["word"] for item in sampled_data]

    return random_words

@tool
def translate_words(random_words: list, source_language: str, target_language: str) -> dict:
        """
        Translates a list of words from a source language to a target language.
        Leverages an LLM invocation and enforces a strict JSON output shape.

        :param random_words: List of string tokens/words to translate.
        :param source_language: The language the vocabulary words currently belong to.
        :param target_language: The language you want the words translated into.
        :return: A structured Python dictionary containing original and translated pairs.
        """

        # 1. Build a structured multiline instruction string for the translation engine.
        #    Uses double curly braces {{ }} to escape literal JSON syntax inside the f-string.
        prompt = (
            f"You are a precise translation engine.\n"
            f"Translate each of the following {len(random_words)} words from {source_language} to {target_language}.\n"
            f"Return ONLY valid JSON with this exact structure:\n"
            f'{{"translations": [{{"source": "<original>", "target": "<translated>"}}], ...}}\n'
            f"No explanations, no extra fields, no markdown.\n"
            f"Words: {json.dumps(random_words, ensure_ascii=False)}"
        # ensure_ascii=False preserves non-English scripts natively
        )

        # 2. Package the prompt string into a LangChain HumanMessage and send it to the model.
        #    Make sure 'translation_model' is instantiated and available in your script's scope.
        response = translation_model.invoke([HumanMessage(content=prompt)])

        # 3. Safely extract the raw string text content out of the model response object.
        #    Uses getattr as a bulletproof fallback in case response structure varies.
        text = getattr(response, "content", str(response))

        # 4. JSON parsing pipeline with a regex extraction fallback strategy.
        try:
            # First pass: Attempt a clean, direct parse assuming the LLM followed instructions perfectly.
            parsed = json.loads(text)
        except Exception:
            import re
            # Fallback pass: If the LLM wrapped its reply in markdown text or extra descriptions,
            # use a regular expression to isolate the first complete curly brace { ... } JSON block.
            # re.DOTALL ensures that the dot (.) character captures line breaks seamlessly.
            match = re.search(r"\{.*\}", text, re.DOTALL)

            # If a match is found, parse the captured string subset; otherwise, fall back to an empty dict.
            parsed = json.loads(match.group(0)) if match else {}

            translation_list = parsed.get("translations", [])
            model_map = {item.get("source"): item.get("target") for item in translation_list if isinstance(item, dict)}

            # Ensure we return translations in the same order as input; fall back to identity if missing
            ordered_translations = [
                {"source": w, "target": model_map.get(w, model_map.get(w.capitalize(), w))}
                for w in random_words
            ]

            # 5. Package the ordered translation results into the final payload structure.
            #    This returns the expected schema back to the LangGraph node.
            return {
                "translations": ordered_translations,
                "source_language": source_language,
                "target_language": target_language
            }
