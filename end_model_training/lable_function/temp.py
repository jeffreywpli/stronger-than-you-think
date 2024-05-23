



if __name__ == "__main__":
    text_to_translate = "Hello, how are you?"
    destination_language = "zh-cn"  # Japanese language code

    translated_text = translate_text(text_to_translate, destination_language)
    print(f"Translated Text: {translated_text}")
    
from googletrans import LANGUAGES

def list_languages():
    for code, language in LANGUAGES.items():
        print(f"{code}: {language}")

if __name__ == "__main__":
    list_languages()


# pip install nltk
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return set(synonyms)

# Example usage
word = "happy"
synonyms = get_synonyms(word)
print(synonyms)
