import os
import json
import random

from langchain_core.tools  import tool

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
    random_keys = random.sample(list(words_filtered_by_difficulty.keys()), safe_n)
    random_word_dict = {k: words_filtered_by_difficulty[k] for k in random_keys}

    # 5. Extract the string value token to return clean vocabulary words
    random_words = [item["word"] for item in random_word_dict.values()]

    return random_words