import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler


dfM= pd.read_csv('movies.csv')
dfU = pd.read_csv('user_reviews.csv')
df = pd.merge(dfM, dfU, on='movieId', how='outer') 
columns_to_drop = [
    'userId', 'movieId', 'movieURL', 'movieRank', 'reviewId', 'movieTitle',
    'isVerified', 'isSuperReviewer', 'hasProfanity', 'movieYear',
    'userRealm', 'creationDate', 'userDisplayName','hasSpoilers', 'score'
]
df.drop(columns_to_drop, axis=1, inplace=True)
# Remove the '%' symbol and convert to float
df['audience_score'] = df['audience_score'].str.rstrip('%').astype(float)
df['critic_score'] = df['critic_score'].str.rstrip('%').astype(float)
df = df.dropna()
def create_targetR(rating):
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'
def create_targetS(rating):
    if rating >= 60:
        return 'positive'
    elif rating <= 40:
        return 'negative'
    else:
        return 'neutral'

df['TargetR'] = df['rating'].apply(create_targetR)
df['TargetA'] = df['audience_score'].apply(create_targetS)
df['TargetC'] = df['critic_score'].apply(create_targetS)
rating = df.TargetR.value_counts()
critic = df.TargetC.value_counts()
audience = df.TargetA.value_counts()
min_class_count = df['TargetR'].value_counts()['negative']

# Step 2: Separate the classes
df_negative = df[df['TargetR'] == 'negative']
df_positive = df[df['TargetR'] == 'positive'].sample(n=min_class_count, random_state=42)
df_neutral = df[df['TargetR'] == 'neutral'].sample(n=min_class_count, random_state=42)

# Step 3: Combine them into a balanced dataset
bdf = pd.concat([df_negative, df_positive, df_neutral], ignore_index=True)

# Optional: Shuffle the resulting DataFrame
bdf = bdf.sample(frac=1, random_state=42).reset_index(drop=True)
import re
import string
import nltk
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    text = text.lower()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    text = re.sub(r'[\d{}]'.format(re.escape(string.punctuation)), ' ', text)

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)

bdf['cleaned_quote'] = bdf['quote'].astype(str).apply(clean_text)
le = LabelEncoder()
bdf['F_Target'] = le.fit_transform(bdf['TargetR'])
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

negation_words = [
    'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 
    'nor', 'nowhere', 'hardly', 'scarcely', 'barely', 'doesn\'t', 
    'isn\'t', 'wasn\'t', 'shouldn\'t', 'wouldn\'t', 'couldn\'t', 
    'won\'t', 'can\'t', 'don\'t', 'didn\'t', 'hasn\'t', 'haven\'t', 'hadn\'t'
]

custom_stop_words = list(ENGLISH_STOP_WORDS.difference(negation_words))

# 2. Features and Labels
X = bdf['cleaned_quote']
y = bdf['F_Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=bdf['F_Target']
)

# Define pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=custom_stop_words)),
    ('logreg', LogisticRegression(max_iter=2000))
])

# Define parameter grid
param_grid = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1,3)],
    'tfidf__min_df': [1, 3, 5],
    'tfidf__max_df': [0.9, 0.95],
    'logreg__C': np.linspace(0.1,2.1,10),  # Regularization strength
    'logreg__penalty': ['l2'],
    'logreg__solver': ['lbfgs']  # Suitable for multiclass
}

# Setup GridSearch
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=3, 
    verbose=2, 
    n_jobs=-1,
    scoring='f1_macro'  # or 'accuracy', 'f1_weighted'
)

# Run grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Params:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Evaluate on test set
y_pred_best = grid_search.best_estimator_.predict(X_test)

y_train_pred = grid_search.best_estimator_.predict(X_train)

from sklearn.metrics import classification_report
# Test predictions (already computed as y_pred)
print("=== Train Set Performance ===")
print(classification_report(y_train, y_train_pred, target_names=le.classes_))

print("=== Test Set Performance ===")
print(classification_report(y_test, y_pred_best, target_names=le.classes_))
# 8. Save model
import pickle
with open("model1.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)

