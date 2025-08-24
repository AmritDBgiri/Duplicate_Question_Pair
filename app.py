import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only the first run will download)
nltk.download('stopwords')
STOP_WORDS = stopwords.words("english")

# -------------------------------
# Helper Functions
# -------------------------------

def test_common_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1) + len(w2)

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    return token_features

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 2
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2
    return length_features

def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', ' percent').replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ').replace('€', ' euro ').replace('@', ' at ')
    q = q.replace('[math]', '').replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ').replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = {"can't": "can not", "won't": "will not", "n't": " not", "'re": " are", "'ll": " will", "'ve": " have"}
    q_decontracted = []
    for word in q.split():
        if word in contractions:
            word = contractions[word]
        q_decontracted.append(word)
    q = ' '.join(q_decontracted)

    q = re.sub(r"<.*?>", "", q)
    q = re.sub(r"\W", " ", q).strip()
    return q

def query_point_creator(q1, q2, cv):
    input_query = []
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    input_query.append(len(q1))
    input_query.append(len(q2))
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    token_features = test_fetch_token_features(q1, q2)
    input_query.extend(token_features)

    length_features = test_fetch_length_features(q1, q2)
    input_query.extend(length_features)

    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()
    return np.hstack((np.array(input_query).reshape(1, 17), q1_bow, q2_bow))

# -------------------------------
# Load Model & Vectorizer
# -------------------------------
try:
    model = pickle.load(open("model.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please check 'model.pkl' and 'cv.pkl'.")
    st.stop()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Duplicate Question Detector")

q1 = st.text_input("Enter question 1")
q2 = st.text_input("Enter question 2")

if st.button("Find"):
    if q1 and q2:
        try:
            query = query_point_creator(q1, q2, cv)
            result = model.predict(query)[0]
            if result:
                st.success("Duplicate")
            else:
                st.info("Not Duplicate")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("⚠️ Please enter both questions.")
     