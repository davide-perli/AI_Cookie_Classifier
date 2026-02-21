import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



rows = []
with open('classified_cookies.csv', 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        parts = line.strip().split(',', 6)
        # Skip id (just a counter)
        row = parts[1:] if len(parts) > 1 else []
        # Pad with empty strings if row has fewer than 6 columns
        while len(row) < 6:
            row.append('')
        # Only keep first 6 columns
        rows.append(row[:6])

df = pd.DataFrame(rows, columns=['Cookie_Name', 'Category', 'Provider', 'Site_Found', 'Duration', 'Description'])
df['Category'] = df['Category'].replace({'Functional': 'Preferences'}) # Apperently I have corrupt data
print(f"Loaded {len(df)} rows from CSV")

valid_categories = ['Marketing', 'Statistics', 'Necessary', 'Preferences']
df = df[df['Category'].isin(valid_categories)]
print(f"After filtering for valid categories: {len(df)} rows")
print(f"Category distribution:\n{df['Category'].value_counts()}\n")

# Split data: 90% training, 10% testing
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

X_train = train_df[['Cookie_Name', 'Provider', 'Site_Found', 'Duration']]
y_train = train_df['Category']  # Labels are the Category column
# mask = np.random.rand(len(X_train)) < 0.5
# X_train.loc[mask, 'Description'] = ''

X_test = test_df[['Cookie_Name', 'Provider', 'Site_Found', 'Duration']].copy()
# X_test['Description'] = ''
y_test = test_df['Category']  

print(f"Training data: {len(X_train)} rows")
print(f"Training labels (Category): {len(y_train)} rows")
print(f"Testing data: {len(X_test)} rows\n")

def clean_domain(text):
    if pd.isna(text):
        return ''
    text = text.replace('.', ' ')
    text = text.replace('-', ' ')
    return text


def combine_features(df):
    return (
        df['Cookie_Name'].fillna('') + ' ' +
        df['Provider'].apply(clean_domain) + ' ' +
        df['Site_Found'].apply(clean_domain) + ' ' +
        df['Duration'].fillna('')# + ' ' +
        # df['Description'].fillna('')
    )

X_train_text = combine_features(X_train)
X_test_text = combine_features(X_test)

print("Example train text:")
print(X_train_text.iloc[0])

print("\nExample test text:")
print(X_test_text.iloc[0])

tfidf = TfidfVectorizer(
    max_features=60000,   # Limit memory
    ngram_range=(1,2),     # Words + word pairs
    min_df=10,              # Ignore rare junk
    sublinear_tf=True
)

X_train_encoded = tfidf.fit_transform(X_train_text)

X_test_encoded = tfidf.transform(X_test_text)

print("TF-IDF shape:", X_train_encoded.shape)



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
        if i < len(conf_matrix):
            correct = conf_matrix[i, i]
            total = conf_matrix[i, :].sum()
            acc = (correct / total * 100) if total > 0 else 0
            print(f"  {cat}: {correct}/{total} = {acc:.2f}%")

    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {model_name} Test Accuracy: {test_acc*100:.2f}%')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.pdf'
    plt.savefig(filename, format='pdf')
    print(f"Saved confusion matrix to: {filename}")
    
    plt.show()

# print("\n" + "="*80)
# print("KNN")
# print("="*80)
# knn = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights='distance', algorithm='brute')
# knn.fit(X_train_encoded, y_train)
# model_train_preds = knn.predict(X_train_encoded)
# test_preds = knn.predict(X_test_encoded)
# evaluate_model("KNN", y_test, test_preds, model_train_preds)

# print("\n" + "="*80)
# print("KNN WITH INVERSE CLASS WEIGHTING")
# print("="*80)

# # Compute inverse class weights for neighbors
# class_weights_inv = compute_sample_weight('balanced', y_train)
# y_train_with_weights = y_train.copy()

# # Create weighted KNN-like predictions where nearby samples are weighted by inverse class frequency
# distances, indices = knn.kneighbors(X_test_encoded)

# # Inverse of class frequencies as weights
# y_unique = y_train.unique()
# class_freq = y_train.value_counts()
# class_weight_dict = {}
# for cls in y_unique:
#     class_weight_dict[cls] = 1.0 / (class_freq[cls] / len(y_train))

# weighted_preds = []
# for i in range(len(X_test_encoded)):
#     neighbor_indices = indices[i]
#     neighbor_distances = distances[i]
#     neighbor_labels = y_train.iloc[neighbor_indices].values
#     neighbor_weights = np.array([class_weight_dict[label] for label in neighbor_labels])
    
#     # Weight by both distance (closer = more important) and inverse class frequency
#     distance_weights = 1.0 / (neighbor_distances + 1e-8)
#     combined_weights = distance_weights * neighbor_weights
    
#     # Normalize weights
#     combined_weights = combined_weights / combined_weights.sum()
    
#     # Score each class
#     class_scores = {}
#     for cls in y_unique:
#         class_scores[cls] = combined_weights[neighbor_labels == cls].sum()
    
#     # Predict the class with highest score
#     pred = max(class_scores, key=class_scores.get)
#     weighted_preds.append(pred)

# weighted_preds = np.array(weighted_preds)
# train_preds_weighted = knn.predict(X_train_encoded)
# evaluate_model("KNN (Class-Weighted Neighbors)", y_test, weighted_preds, train_preds_weighted)


# Heavier weights for higher-priority classes
class_weight_map = {
    'Necessary': 5.0,      
    'Preferences': 2.0,
    'Statistics': 1.5,
    'Marketing': 1.0,    
}      
sample_weight = y_train.map(class_weight_map).fillna(1.0)

print("Class weights:", class_weight_map)

lg_regerssion = LogisticRegression(
    max_iter=10000,
    solver='saga',  
    random_state=0
)
lg_regerssion.fit(X_train_encoded, y_train, sample_weight=sample_weight)
model_train_preds = lg_regerssion.predict(X_train_encoded)
y_lg_pred = lg_regerssion.predict(X_test_encoded)
evaluate_model("Logistic Regression (Manual Weights)", y_test, y_lg_pred, model_train_preds)

print("\n" + "="*80)
print("TRYING: Logistic Regression with class_weight='balanced'")
print("="*80)

lg_balanced = LogisticRegression(
    max_iter=10000,
    solver='saga',
    random_state=0,
    class_weight='balanced'
)
lg_balanced.fit(X_train_encoded, y_train)
model_train_preds_bal = lg_balanced.predict(X_train_encoded)
y_lg_pred_bal = lg_balanced.predict(X_test_encoded)
evaluate_model("Logistic Regression (Balanced)", y_test, y_lg_pred_bal, model_train_preds_bal)

print("\n" + "="*80)
print("TRYING: LinearSVC")
print("="*80)

svc = LinearSVC(
    class_weight={
        'Necessary': 5.0,
        'Preferences': 2.0,
        'Statistics': 1.5,
        'Marketing': 1.0
    },
    max_iter=10000
)

svc.fit(X_train_encoded, y_train)

model_train_preds_svc = svc.predict(X_train_encoded)
y_svc_pred = svc.predict(X_test_encoded)

evaluate_model("LinearSVC", y_test, y_svc_pred, model_train_preds_svc)


# print("\n" + "="*80)
# print("TRYING: RandomForest with class_weight='balanced_subsample'")
# print("="*80)


# rf = RandomForestClassifier(
#     n_estimators=100,
#     random_state=0,
#     n_jobs=-1,
#     class_weight='balanced_subsample',
#     max_depth=20,
#     max_features='sqrt'   
# )

# rf = RandomForestClassifier(
#     n_estimators=50,
#     max_depth=15,
#     max_features=0.1,
#     n_jobs=-1,
#     class_weight='balanced_subsample'
# )

# rf = RandomForestClassifier(
#     n_estimators=50,              # Fewer trees = much faster
#     max_depth=15,                # Prevent huge trees
#     max_features=0.05,           # Only use 5% of features per split
#     min_samples_split=10,        # Avoid tiny splits
#     min_samples_leaf=5,          # Prevent overfitting
#     n_jobs=-1,
#     random_state=0,
#     class_weight='balanced_subsample'
# )

# rf.fit(X_train_encoded, y_train)
# model_train_preds_rf = rf.predict(X_train_encoded)
# y_rf_pred = rf.predict(X_test_encoded)
# evaluate_model("RandomForest", y_test, y_rf_pred, model_train_preds_rf)