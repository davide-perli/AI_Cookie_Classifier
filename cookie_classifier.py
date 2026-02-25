import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, lightgbm as lgb, xgboost as xgb, catboost
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, chi2

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

# print("Example train text:")
# print(X_train_text.iloc[0])

# print("\nExample test text:")
# print(X_test_text.iloc[0])

tfidf = TfidfVectorizer(
    max_features=80000,   # Limit memory
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
    filename = f'confusion_matrix_{model_name.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    plt.savefig(filename, format='png')
    print(f"Saved confusion matrix to: {filename}")
    
    # plt.show()

# print("\n" + "="*80)
# print("KNN")
# print("="*80)
# # 62,94%
# knn = KNeighborsClassifier(n_neighbors=4, metric='cosine', weights='distance', algorithm='brute')
# knn.fit(X_train_encoded, y_train)
# model_train_knn_preds = knn.predict(X_train_encoded)
# test_knn_preds = knn.predict(X_test_encoded)
# evaluate_model("KNN", y_test, test_knn_preds, model_train_knn_preds)

# print("\n" + "="*80)
# print("KNN WITH INVERSE CLASS WEIGHTING")
# print("="*80)

# knn = KNeighborsClassifier(n_neighbors=4, metric='cosine', weights='distance', algorithm='brute')
# knn.fit(X_train_encoded, y_train)

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

# weighted_knnww_preds = []
# for i in range(X_test_encoded.shape[0]):
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
#     weighted_knnww_preds.append(pred)

# weighted_knnww_preds = np.array(weighted_knnww_preds)
# train_preds_knnww_weighted = knn.predict(X_train_encoded)
# evaluate_model("KNN WITH INVERSE CLASS WEIGHTING", y_test, weighted_knnww_preds, train_preds_knnww_weighted)

# print("\n" + "="*80)
# print("TRYING: Logistic Regression with manual weights")
# print("="*80)

# # 95,52%

# class_weight_map = {
#     'Necessary': 5.0,      
#     'Preferences': 2.0,
#     'Statistics': 1.5,
#     'Marketing': 1.0,    
# }      
# sample_weight = y_train.map(class_weight_map).fillna(1.0)

# lg_regerssion = LogisticRegression(
#     l1_ratio=0.5,
#     C=1.0,
#     max_iter=10000,
#     solver='saga',  
#     random_state=0
# )
# lg_regerssion.fit(X_train_encoded, y_train, sample_weight=sample_weight)
# model_train_preds = lg_regerssion.predict(X_train_encoded)
# y_lg_pred = lg_regerssion.predict(X_test_encoded)
# evaluate_model("Logistic Regression (Manual Weights)", y_test, y_lg_pred, model_train_preds)

# print("\n" + "="*80)
# print("TRYING: Logistic Regression with class_weight='balanced'")
# print("="*80)

# # 93,72 %

# lg_balanced = LogisticRegression(
#     max_iter=10000,
#     solver='saga',
#     random_state=0,
#     class_weight='balanced'
# )
# lg_balanced.fit(X_train_encoded, y_train)
# model_train_preds_bal = lg_balanced.predict(X_train_encoded)
# y_lg_pred_bal = lg_balanced.predict(X_test_encoded)
# evaluate_model("Logistic Regression (Balanced)", y_test, y_lg_pred_bal, model_train_preds_bal)

# print("\n" + "="*80)
# print("TRYING: SGD Classifier")
# print("="*80)

# # 97,62%

# sgdc = SGDClassifier(
#     loss='hinge', 
#     penalty='l2',
#     alpha=1e-7,
#     max_iter=1000,
#     n_jobs=-1,
#     random_state=0,
#     learning_rate='optimal',
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=50,
#     class_weight={"Necessary": 3.0, "Preferences": 2.5, "Statistics": 1.0, "Marketing": 1.0}
# )

# sgdc.fit(X_train_encoded, y_train)
# sgdc_train_pred = sgdc.predict(X_train_encoded)
# sgdc_test_pred = sgdc.predict(X_test_encoded)
# evaluate_model("SGD Classifier", y_test, sgdc_test_pred, sgdc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Passive Aggressive Classifier")
# print("="*80)

# # 97,35%

# pac = SGDClassifier(
#     loss='hinge', 
#     penalty='l2',
#     alpha=1e-7,
#     max_iter=1000,
#     n_jobs=-1,
#     random_state=0,
#     learning_rate='pa1',
#     eta0=0.8,
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=50,
#     class_weight={"Necessary": 3.0, "Preferences": 2.5, "Statistics": 1.0, "Marketing": 1.0}
# )

# pac.fit(X_train_encoded, y_train)
# pac_train_pred = pac.predict(X_train_encoded)
# pac_test_pred = pac.predict(X_test_encoded)
# evaluate_model("Passive Aggressive Classifier", y_test, pac_test_pred, pac_train_pred)

# print("\n" + "="*80)
# print("TRYING: Linear SVC")
# print("="*80)

# # 97,22%

# svc = LinearSVC(
#     C=1.3,
#     class_weight='balanced',
#     max_iter=20000,
#     random_state=0,
#     penalty='l1'
# )

# svc.fit(X_train_encoded, y_train)

# model_train_preds_svc = svc.predict(X_train_encoded)
# y_svc_pred = svc.predict(X_test_encoded)

# evaluate_model("LinearSVC", y_test, y_svc_pred, model_train_preds_svc)

# print("\n" + "="*80)
# print("TRYING: Linear SVC With Manual Weights")
# print("="*80)

# # 97,37%

# svc = LinearSVC(
#     C=0.6,
#     class_weight={
#         'Necessary': 3.5,
#         'Preferences': 2.7,
#         'Statistics': 1.7,
#         'Marketing': 1.0
#     },
#     max_iter=20000,
#     random_state=0,
#     penalty='l1'
# )

# svc.fit(X_train_encoded, y_train)

# train_pred = svc.predict(X_train_encoded)
# test_pred = svc.predict(X_test_encoded)

# evaluate_model(f"LinearSVC Manual Weights", y_test, test_pred, train_pred)


# print("\n" + "="*80)
# print("TRYING: RandomForest with class_weight='balanced_subsample'")
# print("="*80)

# # 92,02%

# rf = RandomForestClassifier(
#     n_estimators=300,
#     random_state=0,
#     n_jobs=-1,
#     class_weight='balanced_subsample',
#     max_depth=None,
#     max_features='sqrt'   
# )

# rf.fit(X_train_encoded, y_train)
# model_train_preds_rf = rf.predict(X_train_encoded)
# y_rf_pred = rf.predict(X_test_encoded)
# evaluate_model("RandomForest Classifier", y_test, y_rf_pred, model_train_preds_rf)

# print("\n" + "="*80)
# print("TRYING: Decision Tree Classifier")
# print("="*80)

# # 98,17%

# dtc = DecisionTreeClassifier(
#     criterion='gini',
#     max_depth=None,
#     max_features=None,
#     class_weight={"Necessary": 3.5, "Preferences": 2.7, "Statistics": 1.7, "Marketing": 1.0},
#     random_state=0
# )

# dtc.fit(X_train_encoded, y_train)
# dtc_train_pred = dtc.predict(X_train_encoded)
# dtc_test_pred = dtc.predict(X_test_encoded)
# evaluate_model("Decision Tree Classifier", y_test, dtc_test_pred, dtc_train_pred)

# # NEED FURTHER TUNING (current 21,65%)
# print("\n" + "="*80)
# print("TRYING: Ada Boost Classifier")
# print("="*80)

# ada_class_weight_map = {
#     'Necessary': 1.6,
#     'Preferences': 1.4,
#     'Statistics': 1.1,
#     'Marketing': 1.0,
# }
# ada_sample_weight = y_train.map(ada_class_weight_map).fillna(1.0)

# ada_base_tree = DecisionTreeClassifier(
#     max_depth=6,
#     min_samples_leaf=5,
#     class_weight='balanced',
#     random_state=0
# )

# abc = AdaBoostClassifier(
#     estimator=ada_base_tree,
#     n_estimators=400,
#     learning_rate=0.5,
#     random_state=0
# )

# abc.fit(X_train_encoded, y_train, sample_weight=ada_sample_weight)
# abc_train_pred = abc.predict(X_train_encoded)
# abc_test_pred = abc.predict(X_test_encoded)
# evaluate_model("Ada Boost Classifier", y_test, abc_test_pred, abc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Multinomial Naive Bayes")
# print("="*80)

# # 93,98%

# selector_mnb = SelectKBest(chi2, k=18500)
# X_train_small_mnb = selector_mnb.fit_transform(X_train_encoded, y_train)
# X_test_small_mnb = selector_mnb.transform(X_test_encoded)

# mnb = MultinomialNB(alpha=1e-9)
# mnb.fit(X_train_small_mnb, y_train)
# mnb_train_pred = mnb.predict(X_train_small_mnb)
# mnb_test_pred = mnb.predict(X_test_small_mnb)
# evaluate_model("Multinomial Naive Bayes", y_test, mnb_test_pred, mnb_train_pred)

# print("\n" + "="*80)
# print("TRYING: Complement Naive Bayes")
# print("="*80)
# # 93,93 %%

# selector_cnb = SelectKBest(chi2, k=17000)
# X_train_small_cnb = selector_cnb.fit_transform(X_train_encoded, y_train)
# X_test_small_cnb = selector_cnb.transform(X_test_encoded)

# cnb = ComplementNB(alpha=1e-8)
# cnb.fit(X_train_small_cnb, y_train)
# cnb_train_pred = cnb.predict(X_train_small_cnb)
# cnb_test_pred = cnb.predict(X_test_small_cnb)
# evaluate_model("Complement Naive Bayes", y_test, cnb_test_pred, cnb_train_pred)

# print("\n" + "="*80)
# print("TRYING: Stacking Classifier")
# print("="*80)

# base_estimators = [
#     ("lr", LogisticRegression(
#         max_iter=3000,
#         solver="saga",
#         class_weight="balanced",
#         random_state=0
#     )),
#     ("sgd", SGDClassifier(
#         loss="log_loss",
#         alpha=1e-5,
#         class_weight="balanced",
#         max_iter=2000,
#         tol=1e-3,
#         n_jobs=-1,
#         random_state=0
#     )),
#     ("cnb", ComplementNB(alpha=0.1))
# ]

# meta_model = LogisticRegression(
#     max_iter=2000,
#     solver="lbfgs",
#     class_weight="balanced",
#     random_state=0
# )

# stack_clf = StackingClassifier(
#     estimators=base_estimators,
#     final_estimator=meta_model,
#     cv=3,
#     n_jobs=-1,
#     passthrough=False
# )

# stack_clf.fit(X_train_encoded, y_train)
# stack_train_pred = stack_clf.predict(X_train_encoded)
# stack_test_pred = stack_clf.predict(X_test_encoded)
# evaluate_model("Stacking Classifier", y_test, stack_test_pred, stack_train_pred)

# print("\n" + "="*80)
# print("TRYING: Voting Classifier")
# print("="*80)

# voting_clf = VotingClassifier(
#     estimators=[
#         ("lr", LogisticRegression(
#             max_iter=3000,
#             solver="saga",
#             class_weight="balanced",
#             random_state=0
#         )),
#         ("sgd", SGDClassifier(
#             loss="log_loss",
#             alpha=1e-5,
#             class_weight="balanced",
#             max_iter=2000,
#             tol=1e-3,
#             random_state=0
#         )),
#         ("cnb", ComplementNB(alpha=0.1))
#     ],
#     voting="hard",   
#     n_jobs=-1
# )

# voting_clf.fit(X_train_encoded, y_train)
# voting_train_pred = voting_clf.predict(X_train_encoded)
# voting_test_pred = voting_clf.predict(X_test_encoded)
# evaluate_model("Voting Classifier", y_test, voting_test_pred, voting_train_pred)

# print("\n" + "="*80)
# print("TRYING: MLP Classifier")
# print("="*80)

# mlpc = MLPClassifier(
#     hidden_layer_sizes=(200,),   
#     activation='relu',
#     solver='adam',              
#     alpha=1e-5,                 
#     batch_size=512,              
#     learning_rate_init=0.001,
#     max_iter=200,
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=5,
#     random_state=42
# )

# mlp_label_encoder = LabelEncoder()
# y_train_mlp = mlp_label_encoder.fit_transform(y_train)

# mlpc.fit(X_train_encoded, y_train_mlp)
# mlpc_train_pred_int = mlpc.predict(X_train_encoded)
# mlpc_test_pred_int = mlpc.predict(X_test_encoded)
# mlpc_train_pred = mlp_label_encoder.inverse_transform(mlpc_train_pred_int)
# mlpc_test_pred = mlp_label_encoder.inverse_transform(mlpc_test_pred_int)
# evaluate_model("MLP Classifier", y_test, mlpc_test_pred, mlpc_train_pred)

# print("\n" + "="*80)
# print("TRYING: LightGBM Classifier")
# print("="*80)
# # 98,47%
# lgbm = lgb.LGBMClassifier(
#     num_iterations=700,
#     num_leaves=120,
#     random_state=0,
#     n_jobs=-1,
#     # reg_alpha=0.2,
#     # reg_lambda=0.2,
#     force_col_wise=True
# )

# lgbm.fit(X_train_encoded, y_train, eval_set=[(X_test_encoded, y_test)], eval_metric='multi_logloss', callbacks=[lgb.early_stopping(30)])
# lgbm_train_pred = lgbm.predict(X_train_encoded)
# lgbm_test_pred = lgbm.predict(X_test_encoded)
# evaluate_model("LightGBM Classifier", y_test, lgbm_test_pred, lgbm_train_pred)

print("\n" + "="*80)
print("TRYING: XGBoost Classifier")
print("="*80)

# 97,18%

selector_xgb = SelectKBest(chi2, k=8000)
X_train_small_xgb = selector_xgb.fit_transform(X_train_encoded, y_train)
X_test_small_xgb = selector_xgb.transform(X_test_encoded)

xgb_label_encoder = LabelEncoder()
y_train_xgb_full = xgb_label_encoder.fit_transform(y_train)
y_test_xgb = xgb_label_encoder.transform(y_test)

X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
    X_train_small_xgb,
    y_train_xgb_full,
    test_size=0.1,
    random_state=42
)

xgb_clf = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(xgb_label_encoder.classes_),
    n_estimators=300,
    max_depth=8,
    # learning_rate=0.2,
    early_stopping_rounds=20,
    tree_method='hist',
    random_state=0,
    n_jobs=-1,
    eval_metric='mlogloss'
)

xgb_clf.fit(
    X_train_xgb,
    y_train_xgb,
    eval_set=[(X_val_xgb, y_val_xgb)],
    verbose=30
)

xgb_train_pred_int = xgb_clf.predict(X_train_small_xgb)
xgb_test_pred_int = xgb_clf.predict(X_test_small_xgb)

xgb_train_pred = xgb_label_encoder.inverse_transform(xgb_train_pred_int)
xgb_test_pred = xgb_label_encoder.inverse_transform(xgb_test_pred_int)

evaluate_model("XGBoost Classifier", y_test, xgb_test_pred, xgb_train_pred)


# RidgeClassifier / RidgeClassifierCV (very strong, fast linear baseline for sparse TF-IDF)
# Perceptron (simple linear model, sometimes surprisingly competitive)
# NearestCentroid (cheap baseline; good as a sanity check)
# BernoulliNB (can work well if you binarize features)
# CalibratedClassifierCV(LinearSVC) (if you want better class probabilities from SVM-like performance)


# print("\n" + "="*80)
# print("TRYING: CatBoost Classifier")
# print("="*80)

# # 97,63%

# selector_cat = SelectKBest(chi2, k=8000)
# X_train_small_cat = selector_cat.fit_transform(X_train_encoded, y_train)
# X_test_small_cat = selector_cat.transform(X_test_encoded)

# X_train_cat, X_val_cat, y_train_cat, y_val_cat = train_test_split(
#     X_train_small_cat,
#     y_train,
#     test_size=0.1,
#     random_state=42
# )
# class_weights = {
#     'Necessary': 3.0,
#     'Preferences': 2.5,
#     'Statistics': 1.3,
#     'Marketing': 1.0
# }

# catc = catboost.CatBoostClassifier(
#     iterations=2000,
#     learning_rate=0.7,
#     depth=6,
#     loss_function='MultiClass',
#     eval_metric='Accuracy',
#     class_weights=class_weights,
#     random_seed=42,
#     od_type='IncToDec',
#     od_wait=20,
#     use_best_model=True,
#     task_type='CPU',
#     thread_count=-1,
#     verbose=50
# )

# catc.fit(X_train_cat, y_train_cat, eval_set=[(X_val_cat, y_val_cat)])
# catc_train_pred = catc.predict(X_train_small_cat)
# catc_test_pred = catc.predict(X_test_small_cat)
# evaluate_model("CatBoost Classifier", y_test, catc_test_pred, catc_train_pred)