import re
import emoji
import contractions
from bs4 import BeautifulSoup
import spacy
import pandas as pd
from tqdm import tqdm
from spacy.util import is_package


def get_nlp(model_name="en_core_web_sm"):
    try:
        if not is_package(model_name):
            raise OSError
        return spacy.load(model_name)
    except OSError:
        print(f"spaCy model '{model_name}' not installed, using blank English model instead.")
        return spacy.blank("en")


def extract_features_batch(texts, nlp_model: str = "en_core_web_sm"):
    try:
        spacy.require_gpu()
    except:
        pass

    nlp = get_nlp(nlp_model)
    results = []

    for text in texts:
        text = str(text)

        # ── Step 1: Extract features from RAW text (before any cleaning) ──
        urls     = re.findall(r'http\S+|www\S+|https\S+', text, flags=re.MULTILINE)
        hashtags = re.findall(r'#(\w+)', text)
        mentions = re.findall(r'@(\w+)', text)

        results.append({
            'original': text,
            'urls':          urls,
            'hashtags':      hashtags,
            'mentions':      mentions
        })

    processed_texts = []

    for r in results:
        text = r['original']

        # ── Step 2: HTML unescape ──
        text = BeautifulSoup(text, "html.parser").get_text()

        # ── Step 3: Remove URLs (replace with space to avoid word merging) ──
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)

        # ── Step 4: Replace special boundary chars with a space ──
        #    Pipes, commas, newlines, tabs — anything that separates tokens
        #    must become a space, NOT be deleted silently.
        text = re.sub(r'[\|,\n\r\t]', ' ', text)

        # ── Step 5: Strip # and @ but keep the word they prefix ──
        text = re.sub(r'[#@]', ' ', text)

        # ── Step 6: Remove remaining non-alphanumeric chars (keep sentence punctuation) ──
        text = re.sub(r'[^a-zA-Z0-9\s.!?]', ' ', text)

        # ── Step 7: Lowercase ──
        text = text.lower()

        # ── Step 8: Fix contractions and demojize ──
        text = contractions.fix(text)
        text = emoji.demojize(text, delimiters=(" ", " "))

        # ── Step 9: Collapse multiple spaces ──
        text = re.sub(r'\s+', ' ', text).strip()

        processed_texts.append(text)

    # ── Step 10: spaCy NER pipeline ──
    docs = list(nlp.pipe(processed_texts, batch_size=500, disable=["lemmatizer", "textcat"]))

    for i, doc in enumerate(docs):
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        orgs    = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

        results[i]['persons']       = persons
        results[i]['organizations'] = orgs
        results[i]['has_person']    = 1 if persons else 0
        results[i]['has_org']       = 1 if orgs else 0

        # ── Step 11: Final cleaned_text — letters and spaces only ──
        text = processed_texts[i]
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        results[i]['cleaned_text'] = text

    return results