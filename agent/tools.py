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

    # 3. Use random.sample to pick 'n' unique random keys (IDs) from the word list dictionary.
    #    Then, build a smaller temporary dictionary containing only those selected items.
    random_word_dict = {k: word_list[k] for k in random.sample(list(word_list.keys()) , n)}

    # 4. Extract just the raw text token (the actual string word) from each selected item object.
    random_words = [item["word"] for item in random_word_dict.values()]

    # 5. Send the final list of words back to the calling LangGraph agent state loop.
    return random_words