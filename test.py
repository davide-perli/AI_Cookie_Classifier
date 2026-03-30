import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD

# --- Load CSV ---
rows = []
with open('classified_cookies.csv', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split(',', 6)
        row = parts[1:] if len(parts) > 1 else []
        while len(row) < 6:
            row.append('')
        rows.append(row[:6])

df = pd.DataFrame(rows, columns=['Cookie_Name', 'Category', 'Provider', 'Site_Found', 'Duration', 'Description'])
df['Category'] = df['Category'].replace({'Functional': 'Preferences'})
print(f"Loaded {len(df)} rows from CSV")

# --- Filter valid categories ---
valid_categories = ['Marketing', 'Statistics', 'Necessary', 'Preferences']
df = df[df['Category'].isin(valid_categories)]
print(f"After filtering for valid categories: {len(df)} rows")
print(f"Category distribution:\n{df['Category'].value_counts()}\n")

# --- Split dataset ---
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
X_train = train_df[['Cookie_Name', 'Provider', 'Site_Found', 'Duration']]
y_train = train_df['Category']
X_test = test_df[['Cookie_Name', 'Provider', 'Site_Found', 'Duration']].copy()
y_test = test_df['Category']

print(f"Training data: {len(X_train)} rows")
print(f"Training labels (Category): {len(y_train)} rows")
print(f"Testing data: {len(X_test)} rows\n")

# --- Feature processing ---
def clean_domain(text):
    if pd.isna(text):
        return ''
    text = text.replace('.', ' ').replace('-', ' ')
    return text

def combine_features(df):
    return (
        df['Cookie_Name'].fillna('') + ' ' +
        df['Provider'].apply(clean_domain) + ' ' +
        df['Site_Found'].apply(clean_domain) + ' ' +
        df['Duration'].fillna('')
    )

X_train_text = combine_features(X_train)
X_test_text = combine_features(X_test)

tfidf = TfidfVectorizer(
    max_features=80000,
    ngram_range=(1, 2),
    min_df=10,
    sublinear_tf=True
)

X_train_encoded = tfidf.fit_transform(X_train_text)
X_test_encoded = tfidf.transform(X_test_text)
print("TF-IDF shape:", X_train_encoded.shape)

# --- Evaluation function ---
def evaluate_model(model_name, y_true, y_pred, train_pred):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Training Accuracy: {train_acc*100:.2f}%")
    print(f"{model_name} Test Accuracy: {test_acc*100:.2f}%")

    print(f"\nUnique categories in test set: {sorted(y_true.unique())}")
    print("Test set category distribution:")
    print(y_true.value_counts().sort_index())

    conf_matrix = confusion_matrix(y_true, y_pred)
    categories = sorted(y_true.unique())

    print(f"\nConfusion Matrix ({model_name}):")
    print(f"Categories: {categories}")
    print(conf_matrix)

    print("\nPer-category accuracy:")
    for i, cat in enumerate(categories):
        correct = conf_matrix[i, i]
        total = conf_matrix[i, :].sum()
        acc = (correct / total * 100) if total > 0 else 0
        print(f"  {cat}: {correct}/{total} = {acc:.2f}%")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name} Test Accuracy: {test_acc*100:.2f}%')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    filename_pdf = f'confusion_matrix_{model_name.replace(" ", "_")}.pdf'
    filename_png = f'confusion_matrix_{model_name.replace(" ", "_")}.png'
    plt.savefig(filename_pdf, format='pdf')
    plt.savefig(filename_png, format='png')
    print(f"Saved confusion matrix to: {filename_png}")

# --- MLP Classifier ---
print("\n" + "="*80)
print("TRYING: MLP Classifier")
print("="*80)

mlpc = MLPClassifier(
    hidden_layer_sizes=(200,),
    activation='relu',
    solver='adam',
    alpha=1e-3,
    batch_size=8192,
    learning_rate_init=0.01,
    max_iter=200,
    verbose=10,
    early_stopping=True,
    validation_fraction=0.1,
    beta_1=0.9,
    epsilon=1e-8,
    n_iter_no_change=10,
    random_state=42
)

mlp_label_encoder = LabelEncoder()
y_train_mlp = mlp_label_encoder.fit_transform(y_train)

X_train_encoded_32 = X_train_encoded.astype("float32")
X_test_encoded_32 = X_test_encoded.astype("float32")

mlpc.fit(X_train_encoded_32, y_train_mlp)
mlpc_train_pred_int = mlpc.predict(X_train_encoded_32)
mlpc_test_pred_int = mlpc.predict(X_test_encoded_32)
mlpc_train_pred = mlp_label_encoder.inverse_transform(mlpc_train_pred_int)
mlpc_test_pred = mlp_label_encoder.inverse_transform(mlpc_test_pred_int)

evaluate_model("MLP Classifier", y_test, mlpc_test_pred, mlpc_train_pred)