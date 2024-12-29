import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings(action='ignore')


# Step 1: Preprocess the Lyrics
def preprocess_lyrics(lyrics):
    lyrics = re.sub(r'[^\w\s]', ' ', lyrics).lower()
    return word_tokenize(lyrics)


# Load your dataset
# Example: df = pd.read_csv('lyrics.csv')
# For demonstration, let's create a sample DataFrame
data = {
    "lyrics": [
        "I love you",
        "You love me",
        "I love you",
        "Let’s go crazy",
        "Go crazy",
        "I will always love you"
    ]
}
df = pd.DataFrame(data)
df['processed'] = df['lyrics'].apply(preprocess_lyrics)


# Step 2: Calculate the Self-Similarity Matrix (SSM)
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def compute_ssm(lyrics, measure=levenshtein_distance):
    n = len(lyrics)
    ssm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ssm[i][j] = measure(lyrics[i], lyrics[j])  # Use the chosen similarity measure
    return ssm


# Step 3: Define Other Similarity Measures
def head_similarity(line1, line2):
    return levenshtein_distance(line1[0:2], line2[0:2])


def tail_similarity(line1, line2):
    return levenshtein_distance(line1[-2:], line2[-2:])


def phonetic_similarity(line1, line2):
    # Placeholder: Implement by using CMU Pronunciation dictionary
    # For example: return levenshtein_distance(cmu_dict[line1], cmu_dict[line2])
    return levenshtein_distance(line1, line2)


def pos_similarity(line1, line2):
    # Use NLTK for part-of-speech tagging
    pos1 = nltk.pos_tag(line1)
    pos2 = nltk.pos_tag(line2)
    return levenshtein_distance(pos1, pos2)


def compute_all_ssms(lyrics):
    ssms = {
        'sim_str': compute_ssm(lyrics),
        'sim_head': compute_ssm(lyrics, measure=head_similarity),
        'sim_tail': compute_ssm(lyrics, measure=tail_similarity),
        'sim_phone': compute_ssm(lyrics, measure=phonetic_similarity),  # Add actual phonetic measure
        'sim_pos': compute_ssm(lyrics, measure=pos_similarity),  # Add actual POS measure
        # Add other similarity measures if implemented
    }
    return ssms


# Step 4: Get Word Vectors
word2vec_model = Word2Vec(sentences=df['processed'], vector_size=100, window=5, min_count=1, workers=4)


def get_average_word_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


# Step 5: Prepare Data for CNN
def prepare_data(lyrics_data):
    feature_vectors = []
    for lyric in lyrics_data:
        # Precompute for simplicity, losing a bit of efficiency
        avg_vector = get_average_word_vector(lyric, word2vec_model)
        feature_vectors.append(avg_vector)

    return np.array(feature_vectors)


# Step 7: Define the 1D CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Step 8: Prepare Data for CNN
def prepare_data(lyrics_data):
    feature_vectors = []
    for lyric_lines in lyrics_data:
        avg_vectors = [get_average_word_vector(line.split(), word2vec_model) for line in lyric_lines]
        feature_vectors.append(np.mean(avg_vectors, axis=0))  # Average across lines

    return np.array(feature_vectors)

# Step 7: Using Dummy Data to Train the Model
# For simplicity, we'll simulate some labels (this data isn't realistic!)
df['label'] = [0, 1, 0, 1, 1, 0]  # Dummy binary labels

# Prepare inputs and reshape appropriately for 1D CNN
X = prepare_data(df['processed'])  # Getting feature vectors
y = to_categorical(df['label'])  # Convert labels to categorical

# Reshape X to fit 1D CNN
X = X.reshape(X.shape[0], X.shape[1], 1)  # Shape will be (number of samples, features, 1)

# Step 9: Train the Model
input_shape = (X.shape[1], 1)  # Shape (features, channels)
model = create_cnn_model(input_shape, num_classes=2)
model.fit(X, y, epochs=10, batch_size=2)  # Adjust epochs/batch size as suitable

# Step 10: Evaluate Model (Here we will use the same data for simplicity)
predictions = model.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(df['label'], predicted_classes)
print(f'Accuracy: {accuracy}')
print(predictions)
