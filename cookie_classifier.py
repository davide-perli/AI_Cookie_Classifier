import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, re
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack

CONFUSION_MATRICES_PDF_DIR = Path('confusion_matrices_pdf')
CONFUSION_MATRICES_PNG_DIR = Path('confusion_matrices_png')
DATA_ANALYSIS_PDF_DIR = Path('training_data_analysis_pdf')
DATA_ANALYSIS_PNG_DIR = Path('training_data_analysis_png')
MODEL_ACCURACY_DATA_DIR = Path('model_accuracies')

# hasing pe date

class CookieDataLoader:
    def __init__(self, csv_path='classified_cookies.csv', test_size=0.1, random_state=42):
        self.csv_path = csv_path
        self.test_size = test_size
        self.random_state = random_state
        self._load_and_split()

    def _load_and_split(self):
        rows = []
        with open(self.csv_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                parts = line.strip().split(',', 6) # Skip id (just a counter)
                row = parts[1:] if len(parts) > 1 else []
                while len(row) < 6: # Pad with empty strings if row has fewer than 6 columns
                    row.append('')
                rows.append(row[:6]) # Only keep first 6 columns
        df = pd.DataFrame(rows, columns=['Cookie_Name', 'Category', 'Provider', 'Site_Found', 'Duration', 'Description'])
        df['Category'] = df['Category'].replace({'Functional':'Preferences'})
        valid_categories = ['Marketing','Statistics','Necessary','Preferences']
        df = df[df['Category'].isin(valid_categories)]

        self.df = df
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        self.X_train_raw = train_df[['Cookie_Name','Provider','Site_Found','Duration']]
        self.y_train = train_df['Category']
        self.X_test_raw = test_df[['Cookie_Name','Provider','Site_Found','Duration']].copy()
        self.y_test = test_df['Category']

    @staticmethod
    def combine_features(df):
        def clean_domain(text):
            if pd.isna(text):
                return ''
            return text.replace('.', ' ').replace('-', ' ')
        return (df['Cookie_Name'].fillna('') + ' ' +
                df['Provider'].apply(clean_domain) + ' ' +
                df['Site_Found'].apply(clean_domain) + ' ' +
                df['Duration'].fillna(''))
    
    @staticmethod
    def data_statistics(df):
        DATA_ANALYSIS_PDF_DIR.mkdir(parents=True, exist_ok=True)
        DATA_ANALYSIS_PNG_DIR.mkdir(parents=True, exist_ok=True)
        CookieDataLoader._plot_category_distribution(df)
        CookieDataLoader._plot_category_clusters(df)
        CookieDataLoader._plot_top_cookie_names(df)
        CookieDataLoader._plot_top_words(df)
        CookieDataLoader._plot_top_word_ngrams(df)
        CookieDataLoader._plot_top_char_ngrams(df)
        CookieDataLoader._plot_cookie_wordcloud(df)

    @staticmethod
    def _plot_category_distribution(df):
        plt.figure(figsize=(12,6))

        ax = sns.countplot(
            data=df,
            x='Category',
            hue='Category',
            order=df['Category'].value_counts().index,
            palette='viridis',
            legend=False
        )

        for p in ax.patches:
            ax.annotate(
                f'{int(p.get_height())}',
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center',
                va='bottom'
            )

        plt.title('Distribution of Cookie Categories')
        plt.xlabel('Cookie Category')
        plt.ylabel('Count')

        plt.tight_layout()
        category_distribution_path_pdf = DATA_ANALYSIS_PDF_DIR / f'category_distribution.pdf'
        category_distribution_path_png = DATA_ANALYSIS_PNG_DIR / f'category_distribution.png'
        plt.savefig(category_distribution_path_pdf, format='pdf')
        plt.savefig(category_distribution_path_png, format='png')
        plt.close()

    @staticmethod
    def _plot_category_clusters(df, sample_size=30000):

        df_sample = df.sample(min(sample_size, len(df)), random_state=42)

        corpus = CookieDataLoader.combine_features(df_sample)

        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(corpus)

        svd = TruncatedSVD(n_components=2, random_state=42)
        coords = svd.fit_transform(X)

        df_sample["pca_x"] = coords[:,0]
        df_sample["pca_y"] = coords[:,1]

        plt.figure(figsize=(12,8))

        sns.kdeplot(
            data=df_sample,
            x="pca_x",
            y="pca_y",
            hue="Category",
            fill=True,
            alpha=0.6
        )

        plt.title("Density Clusters of Cookie Categories")
        plt.tight_layout()
        category_clusters_path_pdf = DATA_ANALYSIS_PDF_DIR / f'category_clusters.pdf'
        category_clusters_path_png = DATA_ANALYSIS_PNG_DIR / f'category_clusters.png'
        plt.savefig(category_clusters_path_pdf, format='pdf')
        plt.savefig(category_clusters_path_png, format='png')
        plt.close()

    @staticmethod
    def _plot_top_cookie_names(df, top_n=20):
        top_cookies = df['Cookie_Name'].value_counts().head(top_n)

        plt.figure(figsize=(12,6))

        sns.barplot(
            x=top_cookies.values,
            y=top_cookies.index,
            hue=top_cookies.index,
            palette='viridis',
            legend=False
        )

        plt.title(f'Top {top_n} Most Frequent Cookie Names')
        plt.xlabel('Frequency')
        plt.ylabel('Cookie Name')

        plt.tight_layout()
        top_cookie_names_path_pdf = DATA_ANALYSIS_PDF_DIR / f'top_cookie_names.pdf'
        top_cookie_names_path_png = DATA_ANALYSIS_PNG_DIR / f'top_cookie_names.png'
        plt.savefig(top_cookie_names_path_pdf, format='pdf')
        plt.savefig(top_cookie_names_path_png, format='png')
        plt.close()

    @staticmethod
    def _plot_top_words(df, top_n=20):

        # corpus = CookieDataLoader.combine_features(df)
        corpus = df['Cookie_Name'].dropna().astype(str).str.replace(r"[._\-]", " ", regex=True)

        words = []
        for text in corpus:
            tokens = re.findall(r'\w+', str(text).lower())
            words.extend(tokens)

        top_words = Counter(words).most_common(top_n)
        words_df = pd.DataFrame(top_words, columns=['word','count'])

        plt.figure(figsize=(10,8))

        sns.barplot(
            data=words_df,
            y='word',
            x='count',
            hue='word',
            palette='Blues_r',
            legend=False
        )

        plt.title(f'Top {top_n} Most Frequent Words in Cookies')
        plt.xlabel('Frequency')
        plt.ylabel('Word')

        plt.tight_layout()
        top_words_path_pdf = DATA_ANALYSIS_PDF_DIR / f'top_words.pdf'
        top_words_path_png = DATA_ANALYSIS_PNG_DIR / f'top_words.png'
        plt.savefig(top_words_path_pdf, format='pdf')
        plt.savefig(top_words_path_png, format='png')
        plt.close()

    @staticmethod
    def _plot_top_word_ngrams(df, ngram_range=(2,3), top_n=20):

        # corpus = CookieDataLoader.combine_features(df)
        corpus = df['Cookie_Name'].dropna().astype(str).str.replace(r"[._\-]", " ", regex=True)

        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=ngram_range,
            stop_words='english'
        )

        X = vectorizer.fit_transform(corpus)

        counts = X.sum(axis=0).A1
        ngrams = vectorizer.get_feature_names_out()

        ngram_df = pd.DataFrame({
            "ngram": ngrams,
            "count": counts
        })

        ngram_df = ngram_df.sort_values(by="count", ascending=False).head(top_n)

        # print(f"Word ngrams: {ngram_df}")

        plt.figure(figsize=(10,8))

        sns.barplot(
            data=ngram_df,
            y="ngram",
            x="count",
            hue="ngram",
            palette="mako",
            legend=False
        )

        plt.title(f"Top {top_n} Most Frequent Word ({ngram_range[0]}, {ngram_range[1]})-grams")
        plt.xlabel("Frequency")
        plt.ylabel("N-gram")

        plt.tight_layout()
        top_word_ngrams_path_pdf = DATA_ANALYSIS_PDF_DIR / f'top_word_ngrams.pdf'
        top_word_ngrams_path_png = DATA_ANALYSIS_PNG_DIR / f'top_word_ngrams.png'
        plt.savefig(top_word_ngrams_path_pdf, format='pdf')
        plt.savefig(top_word_ngrams_path_png, format='png')
        plt.close()

# char_vec_mnb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=9, sublinear_tf=True)

    @staticmethod
    def _plot_top_char_ngrams(df, ngram_range=(3,6), top_n=20):

        # corpus = CookieDataLoader.combine_features(df)
        corpus = df['Cookie_Name'].dropna().astype(str).str.replace(r"[._\-]", " ", regex=True)

        vectorizer = CountVectorizer(
            analyzer='char_wb',
            ngram_range=ngram_range
        )

        X = vectorizer.fit_transform(corpus)

        counts = X.sum(axis=0).A1
        ngrams = vectorizer.get_feature_names_out()

        ngram_df = pd.DataFrame({
            "ngram": ngrams,
            "count": counts
        })

        ngram_df = ngram_df.sort_values(by="count", ascending=False).head(top_n)

        # print(f"Char ngrams: {ngram_df}")

        plt.figure(figsize=(10,8))

        sns.barplot(
            data=ngram_df,
            y="ngram",
            x="count",
            hue="ngram",
            palette="mako",
            legend=False
        )

        plt.title(f"Top {top_n} Most Frequent Char ({ngram_range[0]}, {ngram_range[1]})-grams")
        plt.xlabel("Frequency")
        plt.ylabel("N-gram")

        plt.tight_layout()
        top_char_ngrams_path_pdf = DATA_ANALYSIS_PDF_DIR / f'top_char_ngrams.pdf'
        top_char_ngrams_path_png = DATA_ANALYSIS_PNG_DIR / f'top_char_ngrams.png'
        plt.savefig(top_char_ngrams_path_pdf, format='pdf')
        plt.savefig(top_char_ngrams_path_png, format='png')
        plt.close()

    @staticmethod
    def _plot_cookie_wordcloud(df):

        # Combine cookie names into one corpus
        all_text = " ".join(df['Cookie_Name'].dropna().astype(str))

        wc = WordCloud(
            width=900,
            height=450,
            background_color="white",
            colormap="viridis",
            max_words=200
        ).generate(all_text)

        plt.figure(figsize=(14,7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Cookie Names")

        plt.tight_layout()
        cookie_worldcloud_path_pdf = DATA_ANALYSIS_PDF_DIR / f'cookie_worldcloud.pdf'
        cookie_worldcloud_path_png = DATA_ANALYSIS_PNG_DIR / f'cookie_worldcloud.png'
        plt.savefig(cookie_worldcloud_path_pdf, format='pdf')
        plt.savefig(cookie_worldcloud_path_png, format='png')
        plt.close()

class CookieClassifier:
    def __init__(self, loader, model_name="MLP", use_custom_weights=False, max_features=80000, k_best=None, max_features_chars=False, k_best_chars=None):
        self.loader = loader
        self.model_name = model_name
        self.use_custom_weights = use_custom_weights
        self.max_features = None if max_features is True else max_features
        self.k_best = k_best
        # Handle char max_features: True -> None (all char features), False -? skip
        self.max_features_chars = None if max_features_chars is True else max_features_chars
        self.k_best_chars = k_best_chars
        # Prepare text and labels
        self._prepare_text()
        # Vectorize
        self._vectorize()
        # Select top-k features if requested
        self._select_top_features()
        # Vectorize chars if needed
        self._vectorize_chars()
        # Select top features on chars n-grams if needed
        self._select_top_features_chars()
        # Prepare model
        self._prepare_model()
        # Train automatically
        self.train()
        # Evaluate automatically
        self.evaluate()
    
    def _prepare_text(self):
        self.X_train_text = self.loader.combine_features(self.loader.X_train_raw)
        self.X_test_text = self.loader.combine_features(self.loader.X_test_raw)
        self.y_train = self.loader.y_train
        self.y_test = self.loader.y_test

    def _vectorize(self):
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1,2),
            min_df=10,
            sublinear_tf=True,
            dtype=np.float32
        )
        self.X_train_vec = self.tfidf.fit_transform(self.X_train_text)
        self.X_test_vec = self.tfidf.transform(self.X_test_text)

    def _select_top_features(self):
        if self.k_best is not None and self.k_best < self.X_train_vec.shape[1]:
            selector = SelectKBest(chi2, k=self.k_best)
            self.X_train_vec = selector.fit_transform(self.X_train_vec, self.y_train)
            self.X_test_vec = selector.transform(self.X_test_vec)
            self.selector = selector
        else:
            self.selector = None

    def _vectorize_chars(self):
        if self.max_features_chars:
            self.tfidf_chars = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(3,6),
                max_features=self.max_features_chars,
                sublinear_tf=True,
                dtype=np.float32
            )
            self.X_train_chars = self.tfidf_chars.fit_transform(self.X_train_text)
            self.X_test_chars = self.tfidf_chars.transform(self.X_test_text)

            # Apply top-k if requested
            if self.k_best_chars is not None:
                self._select_top_features_chars()

            # Combine with word-level TF-IDF
            self.X_train_vec = hstack([self.X_train_vec, self.X_train_chars])
            self.X_test_vec = hstack([self.X_test_vec, self.X_test_chars])

    def _select_top_features_chars(self):
        if not hasattr(self, 'X_train_chars') or self.X_train_chars is None:
            self.selector_chars = None
            return

        if self.k_best_chars is not None and self.k_best_chars < self.X_train_chars.shape[1]:
            selector = SelectKBest(chi2, k=self.k_best_chars)
            self.X_train_chars = selector.fit_transform(self.X_train_chars, self.y_train)
            self.X_test_chars = selector.transform(self.X_test_chars)
            self.selector_chars = selector
        else:
            self.selector_chars = None

    def _prepare_model(self):
        if self.model_name == "MLP":
            self.model = MLPClassifier(
                    hidden_layer_sizes=(200,),   
                    activation='relu', # tanh has 98,70%
                    solver='adam',              
                    alpha=1e-3,                 
                    batch_size=8192,
                    learning_rate_init=0.01,
                    max_iter=200,
                    verbose=False,
                    early_stopping=True,
                    validation_fraction=0.1,
                    beta_1=0.9,
                    epsilon=1e-8,    
                    n_iter_no_change=10,
                    random_state=42,
            )
        elif self.model_name == "KNN":
            def KNN_custom_weights(distances):
                return (1 / (distances + 1e-5) ** 2.2) * np.exp(-distances)
            
            self.model = KNeighborsClassifier(
                n_neighbors=7,            
                weights=KNN_custom_weights,
                metric='euclidean',     
                algorithm='auto',
                n_jobs=-1
            )
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression(
                max_iter=10000,
                solver='saga',
                random_state=0,
                class_weight='balanced'
            )
        elif self.model_name == "LogisticRegression_MW":
            self.model = LogisticRegression(
                    l1_ratio=0.5,
                    C=1.0,
                    max_iter=10000,
                    solver='saga',  
                    random_state=0
            )
        elif self.model_name == "SGD":
            self.model = SGDClassifier(
                loss='hinge', 
                penalty='l2',
                alpha=1e-7,
                max_iter=1000,
                n_jobs=-1,
                random_state=0,
                learning_rate='optimal',
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                class_weight={"Necessary": 3.0, "Preferences": 2.5, "Statistics": 1.0, "Marketing": 1.0}
            )
        elif self.model_name == "PAC":
            self.model = SGDClassifier(
                loss='hinge', 
                penalty='l2',
                alpha=1e-7,
                max_iter=1000,
                n_jobs=-1,
                random_state=0,
                learning_rate='pa1',
                eta0=0.8,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=50,
                class_weight={"Necessary": 3.0, "Preferences": 2.5, "Statistics": 1.0, "Marketing": 1.0}
            )
        elif self.model_name == "LinearSVC":
            self.model = LinearSVC(
                C=1.3,
                class_weight='balanced',
                max_iter=20000,
                random_state=0,
                penalty='l1'
            )
        elif self.model_name == "LinearSVC_MW":
            self.model = LinearSVC(
                C=0.6,
                max_iter=20000,
                random_state=0,
                penalty='l1',
                class_weight={'Necessary': 3.5, 'Preferences': 2.7, 'Statistics': 1.7, 'Marketing': 1.0},
            )
        elif self.model_name == "RandomForest":
            self.model = RandomForestClassifier(
                n_estimators=300,
                random_state=0,
                n_jobs=-1,
                class_weight='balanced_subsample',
                max_depth=None,
                max_features='sqrt'   
            )
        elif self.model_name == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier(
                criterion='gini',
                max_depth=None,
                max_features=None,
                class_weight={"Necessary": 3.5, "Preferences": 2.7, "Statistics": 1.7, "Marketing": 1.0},
                random_state=0
            )
        elif self.model_name == "AdaBoost":
            ada_base_tree = DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=5,
                random_state=0
            )

            self.model = AdaBoostClassifier(
                estimator=ada_base_tree,
                n_estimators=180,
                learning_rate=0.4,
                random_state=0
            )
        elif self.model_name == "MultinomialNB":
            # Word-level TF-IDF
            self.word_vec_mnb = TfidfVectorizer(ngram_range=(1,2), min_df=4, sublinear_tf=True, dtype=np.float32)
            Xtr_w = self.word_vec_mnb.fit_transform(self.X_train_text)
            Xte_w = self.word_vec_mnb.transform(self.X_test_text)

            # Char-level TF-IDF
            self.char_vec_mnb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=9, sublinear_tf=True, dtype=np.float32)
            Xtr_c = self.char_vec_mnb.fit_transform(self.X_train_text)
            Xte_c = self.char_vec_mnb.transform(self.X_test_text)

            # Feature selection
            k_word = 18500
            k_char = 8000
            self.sel_w_mnb = SelectKBest(chi2, k=k_word)
            self.sel_c_mnb = SelectKBest(chi2, k=k_char)

            Xtr_w_sel = self.sel_w_mnb.fit_transform(Xtr_w, self.y_train)
            Xte_w_sel = self.sel_w_mnb.transform(Xte_w)

            Xtr_c_sel = self.sel_c_mnb.fit_transform(Xtr_c, self.y_train)
            Xte_c_sel = self.sel_c_mnb.transform(Xte_c)

            # Combine word + char features
            self.X_train_vec = hstack([Xtr_w_sel, Xtr_c_sel])
            self.X_test_vec = hstack([Xte_w_sel, Xte_c_sel])

            self.model = MultinomialNB(alpha=1e-9)

            self.sample_weights = compute_sample_weight(
                class_weight={"Necessary": 1.4, "Preferences": 1.3, "Statistics": 1.3, "Marketing": 1.1},
                y=self.y_train
            )
        else:
            raise ValueError(f"Model {self.model_name} not implemented yet")
                
    def _reduce_for_knn(self):
        self.svd = TruncatedSVD(n_components=128, random_state=42)

        self.X_train_vec = self.svd.fit_transform(self.X_train_vec)
        self.X_test_vec = self.svd.transform(self.X_test_vec)

        self.normalizer = Normalizer(copy=False)
        self.X_train_vec = self.normalizer.fit_transform(self.X_train_vec)
        self.X_test_vec = self.normalizer.transform(self.X_test_vec)

    def _reduce_for_adaboost(self):
        
        self.svd_ada = TruncatedSVD(n_components=256, random_state=42)
        self.X_train_vec_ada = self.svd_ada.fit_transform(self.X_train_vec)
        self.X_test_vec_ada = self.svd_ada.transform(self.X_test_vec)

        self.class_weights_ada = {'Necessary': 3.0, 'Preferences': 5.0, 'Statistics': 1.0, 'Marketing': 1.0}

        subset_size = 200000 

        self.X_train_sub, self.y_train_sub = resample(
            self.X_train_vec_ada,
            self.y_train,
            n_samples=subset_size,
            stratify=self.y_train,
            random_state=42
        )

        self.sample_weights_sub = self.y_train_sub.map(self.class_weights_ada).values

    def train(self):
        if self.model_name == "KNN":
            self._reduce_for_knn()
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "LogisticRegression_MW":
            class_weight_map = {'Necessary': 5.0, 'Preferences': 2.0, 'Statistics': 1.5, 'Marketing': 1.0}      
            sample_weight = self.y_train.map(class_weight_map).fillna(1.0)
            self.model.fit(self.X_train_vec, self.y_train, sample_weight=sample_weight)
        elif self.model_name == "AdaBoost":
            self._reduce_for_adaboost()
            self.model.fit(self.X_train_sub, self.y_train_sub, sample_weight=self.sample_weights_sub)
        elif self.model_name == "MultinomialNaiveBayes":
            self.model.fit(self.X_train_vec, self.y_train, sample_weight=self.sample_weights)
        else:
            self.model.fit(self.X_train_vec, self.y_train)

    def evaluate(self):
        train_pred = self.model.predict(self.X_train_vec)
        test_pred = self.model.predict(self.X_test_vec)
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        train_f1 = f1_score(self.y_train, train_pred, average='macro', zero_division=0)
        test_f1 = f1_score(self.y_test, test_pred, average='macro', zero_division=0)
        print(f"{self.model_name} - Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"{self.model_name} - Training F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")

        # Confusion matrix (saved as PDF + PNG, same style as evaluate_model())
        classes = list(getattr(self.model, 'classes_', sorted(pd.unique(pd.concat([self.y_train, self.y_test])))))
        conf_matrix = confusion_matrix(self.y_test, test_pred, labels=classes)

        print(f"\nConfusion Matrix ({self.model_name}):")
        print(f"Categories: {classes}")
        print(conf_matrix)

        print("\nPer-category accuracy:")
        for i, cat in enumerate(classes):
            correct = conf_matrix[i, i]
            total = conf_matrix[i, :].sum()
            acc = (correct / total * 100) if total > 0 else 0
            print(f"  {cat}: {correct}/{total} = {acc:.2f}%")

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(
            f"Confusion Matrix - {self.model_name}\n"
            f"Test Accuracy: {test_acc*100:.2f}%\n"
            f"Test F1 (macro): {test_f1*100:.2f}"
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(self.model_name)).strip('_')

        CONFUSION_MATRICES_PDF_DIR.mkdir(parents=True, exist_ok=True)
        CONFUSION_MATRICES_PNG_DIR.mkdir(parents=True, exist_ok=True)

        pdf_path = CONFUSION_MATRICES_PDF_DIR / f'confusion_matrix_{safe_name}.pdf'
        png_path = CONFUSION_MATRICES_PNG_DIR / f'confusion_matrix_{safe_name}.png'
        plt.savefig(pdf_path, format='pdf')
        plt.savefig(png_path, format='png')
        plt.close()
        print(f"Saved confusion matrix to: {png_path} and {pdf_path}")


data_loader = CookieDataLoader()
CookieDataLoader.data_statistics(data_loader.df)

# classifier = CookieClassifier(
#     loader=data_loader,
#     model_name="LogisticRegression",
#     max_features=50000,
#     k_best=20000,
#     max_features_chars=10000,
#     k_best_chars=5000
# )

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
    max_features=80000,   
    ngram_range=(1,2),     # Words + word pairs
    min_df=10,             
    sublinear_tf=True,
    dtype=np.float32
)

X_train_encoded = tfidf.fit_transform(X_train_text)

X_test_encoded = tfidf.transform(X_test_text)

print("TF-IDF shape:", X_train_encoded.shape)

print(X_train_encoded.dtype)

def evaluate_model(model_name, y_true, y_pred, train_pred, include_f1=True, f1_average="macro"):
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_true, y_pred)
    print(f"\n{model_name} Training Accuracy: {train_acc*100:.2f}%")
    print(f"{model_name} Test Accuracy: {test_acc*100:.2f}%")

    train_f1 = None
    test_f1 = None
    if include_f1:
        train_f1 = f1_score(y_train, train_pred, average=f1_average, zero_division=0)
        test_f1 = f1_score(y_true, y_pred, average=f1_average, zero_division=0)
        print(f"{model_name} Training F1 ({f1_average}): {train_f1*100:.2f}")
        print(f"{model_name} Test F1 ({f1_average}): {test_f1*100:.2f}")

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
    if include_f1 and test_f1 is not None:
        plt.title(
            f"Confusion Matrix - {model_name}\n"
            f"Test Accuracy: {test_acc*100:.2f}%\n"
            f"Test F1 ({f1_average}): {test_f1*100:.2f}"
        )
    else:
        plt.title(f'Confusion Matrix - {model_name} Test Accuracy: {test_acc*100:.2f}%')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', str(model_name)).strip('_')

    CONFUSION_MATRICES_PDF_DIR.mkdir(parents=True, exist_ok=True)
    CONFUSION_MATRICES_PNG_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = CONFUSION_MATRICES_PDF_DIR / f'confusion_matrix_{safe_name}.pdf'
    png_path = CONFUSION_MATRICES_PNG_DIR / f'confusion_matrix_{safe_name}.png'
    plt.savefig(pdf_path, format='pdf')
    plt.savefig(png_path, format='png')
    plt.close()
    print(f"Saved confusion matrix to: {png_path} and {pdf_path}")
    
    # plt.show()

# print("\n" + "="*80)
# print("KNN")
# print("="*80)

# # 95.18%

# svd = TruncatedSVD(n_components=128, random_state=42)
# X_train_reduced = svd.fit_transform(X_train_encoded)
# X_test_reduced = svd.transform(X_test_encoded)

# # Normalize so cosine = dot product

# normalizer = Normalizer(copy=False)
# X_train_reduced = normalizer.fit_transform(X_train_reduced)
# X_test_reduced = normalizer.transform(X_test_reduced)

# def custom_weights(distances):
#     return (1 / (distances + 1e-5) ** 2.2) * np.exp(-distances)

# # 91.67%

# knn = KNeighborsClassifier(
#     n_neighbors=7,            
#     weights=custom_weights,
#     metric='euclidean',       # after normalization == cosine
#     algorithm='auto',
#     n_jobs=-1
# )


# knn.fit(X_train_reduced, y_train)

# model_train_knn_preds = knn.predict(X_train_reduced)
# test_knn_preds = knn.predict(X_test_reduced)

# evaluate_model("KNN", y_test, test_knn_preds, model_train_knn_preds)


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
# print("TRYING: RandomForest")
# print("="*80)

# # 98,18%

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
# model_test_preds_rf = rf.predict(X_test_encoded)
# evaluate_model("RandomForest Classifier", y_test, model_test_preds_rf, model_train_preds_rf)

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

# # NEED FURTHER TUNING (current 94.23%)
print("\n" + "="*80)
print("TRYING: Ada Boost Classifier")
print("="*80)

svd_ada = TruncatedSVD(n_components=300, random_state=42)

X_train_ada = svd_ada.fit_transform(X_train_encoded)
X_test_ada = svd_ada.transform(X_test_encoded)

class_weights = {
    'Necessary': 2.0,
    'Preferences': 1.5,
    'Statistics': 1.0,
    'Marketing': 1.0
}

subset_size = 300000

X_sub, y_sub = resample(
    X_train_ada,
    y_train,
    n_samples=subset_size,
    stratify=y_train,
    random_state=42
)

sample_weights_sub = y_sub.map(class_weights).values

ada_base_tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=15,
    # min_samples_leaf=5,
    random_state=0
)

abc = AdaBoostClassifier(
    estimator=ada_base_tree,
    n_estimators=200,
    learning_rate=0.4,
    random_state=0
)

abc.fit(X_sub, y_sub, sample_weight=sample_weights_sub)

# evaluate on full datasets
abc_train_pred = abc.predict(X_train_ada)
abc_test_pred = abc.predict(X_test_ada)

evaluate_model("Ada Boost Classifier", y_test, abc_test_pred, abc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Multinomial Naive Bayes")
# print("="*80)

# # # 94.27%

# word_vec_mnb = TfidfVectorizer(ngram_range=(1,2), min_df=4, sublinear_tf=True)
# char_vec_mnb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=9, sublinear_tf=True)

# Xtr_w_mnb = (word_vec_mnb.fit_transform(X_train_text)).astype("float32")
# Xte_w_mnb = (word_vec_mnb.transform(X_test_text)).astype("float32")

# Xtr_c_mnb = (char_vec_mnb.fit_transform(X_train_text)).astype("float32")
# Xte_c_mnb = (char_vec_mnb.transform(X_test_text)).astype("float32")

# k_word = 18500         
# k_char = 8000         

# sel_w_mnb = SelectKBest(chi2, k=k_word)
# sel_c_mnb = SelectKBest(chi2, k=k_char)

# Xtr_ws_mnb = sel_w_mnb.fit_transform(Xtr_w_mnb, y_train)
# Xte_ws_mnb = sel_w_mnb.transform(Xte_w_mnb)

# Xtr_cs_mnb = sel_c_mnb.fit_transform(Xtr_c_mnb, y_train)
# Xte_cs_mnb = sel_c_mnb.transform(Xte_c_mnb)

# Xtr_mnb = hstack([Xtr_ws_mnb, Xtr_cs_mnb])
# Xte_mnb = hstack([Xte_ws_mnb, Xte_cs_mnb])

# mnb = MultinomialNB(alpha=1e-9)
# sw = compute_sample_weight(class_weight={"Necessary": 1.4, "Preferences": 1.3, "Statistics": 1.3, "Marketing": 1.1}, y=y_train)
# mnb.fit(Xtr_mnb, y_train, sample_weight=sw)
# mnb_train_predict_label = mnb.predict(Xtr_mnb)
# mnb_test_predict_label = mnb.predict(Xte_mnb)
# evaluate_model("Multinomial Naive Bayes", y_test, mnb_test_predict_label, mnb_train_predict_label)

# print("\n" + "="*80)
# print("TRYING: Complement Naive Bayes")
# print("="*80)

# # # 94.87 %%

# word_vec_cnb = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True)
# char_vec_cnb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=9, sublinear_tf=True)

# Xtr_w_cnb = (word_vec_cnb.fit_transform(X_train_text)).astype("float32")
# Xte_w_cnb = (word_vec_cnb.transform(X_test_text)).astype("float32")

# Xtr_c_cnb = (char_vec_cnb.fit_transform(X_train_text)).astype("float32")
# Xte_c_cnb = (char_vec_cnb.transform(X_test_text)).astype("float32")

# k_word = 50000         
# k_char = 35000         

# sel_w_cnb = SelectKBest(chi2, k=k_word)
# sel_c_cnb = SelectKBest(chi2, k=k_char)

# Xtr_ws_cnb = sel_w_cnb.fit_transform(Xtr_w_cnb, y_train)
# Xte_ws_cnb = sel_w_cnb.transform(Xte_w_cnb)

# Xtr_cs_cnb = sel_c_cnb.fit_transform(Xtr_c_cnb, y_train)
# Xte_cs_cnb = sel_c_cnb.transform(Xte_c_cnb)

# Xtr_cnb = hstack([Xtr_ws_cnb, Xtr_cs_cnb])
# Xte_cnb = hstack([Xte_ws_cnb, Xte_cs_cnb])

# cnb = ComplementNB(alpha=1e-8)
# sw = compute_sample_weight(class_weight={"Necessary": 1.8, "Preferences": 1.4, "Statistics": 1.3, "Marketing": 1.0}, y=y_train)
# cnb.fit(Xtr_cnb, y_train, sample_weight=sw)
# cnb_train_pred = cnb.predict(Xtr_cnb)
# cnb_test_pred = cnb.predict(Xte_cnb)
# evaluate_model("Complement Naive Bayes", y_test, cnb_test_pred, cnb_train_pred)

# print("\n" + "="*80)
# print("TRYING: Bernoulli Naive Bayes")
# print("="*80)

# # 93,75%
# X_train_bin = (X_train_encoded > 0).astype(np.int8)
# X_test_bin = (X_test_encoded > 0).astype(np.int8)

# selector_bnb = SelectKBest(chi2, k=22000) 
# X_train_small_bnb = selector_bnb.fit_transform(X_train_bin, y_train)
# X_test_small_bnb = selector_bnb.transform(X_test_bin)

# bnb = BernoulliNB(alpha=1e-3, binarize=None)  
# bnb.fit(X_train_small_bnb, y_train)

# bnb_train_pred = bnb.predict(X_train_small_bnb)
# bnb_test_pred = bnb.predict(X_test_small_bnb)
# evaluate_model("Bernoulli Naive Bayes", y_test, bnb_test_pred, bnb_train_pred)

# print("\n" + "="*80)
# print("TRYING: Stacking Classifier")
# print("="*80)

# # 95,76

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

# # 96,40%

# voting_clf = VotingClassifier(
#     estimators=[
#         ("lr", LogisticRegression(
#             max_iter=5000,
#             solver="saga",
#             class_weight="balanced",
#             random_state=0
#         )),
#         ("sgd", SGDClassifier(
#             loss="log_loss",
#             alpha=1e-7,
#             max_iter=100,
#             early_stopping=True,
#             validation_fraction=0.1,
#             n_iter_no_change=5,    
#             class_weight={0: 3.0, 1: 2.5, 2: 1.0, 3: 1.0},
#             random_state=0
#         )),
#         ("cnb", ComplementNB(alpha=1e-8))
#     ],
#     voting="hard",   
#     n_jobs=-1,
#     verbose=10
# )

# selector_voting_clf = SelectKBest(chi2, k=10000)
# X_train_small_voting_clf = selector_voting_clf.fit_transform(X_train_encoded, y_train)
# X_test_small_voting_clf = selector_voting_clf.transform(X_test_encoded)

# voting_clf.fit(X_train_small_voting_clf, y_train)
# voting_train_pred = voting_clf.predict(X_train_small_voting_clf)
# voting_test_pred = voting_clf.predict(X_test_small_voting_clf)
# evaluate_model("Voting Classifier", y_test, voting_test_pred, voting_train_pred)

# print("\n" + "="*80)
# print("TRYING: MLP Classifier")
# print("="*80)

# # 98,76%

# mlpc = MLPClassifier(
#     hidden_layer_sizes=(200,),   
#     activation='relu', # tanh has 98,70%
#     solver='adam',              
#     alpha=1e-3,                 
#     batch_size=8192,
#     learning_rate_init=0.01,
#     max_iter=200,
#     verbose=10,
#     early_stopping=True,
#     validation_fraction=0.1,
#     beta_1=0.9,
#     epsilon=1e-8,    
#     n_iter_no_change=10,
#     random_state=42,
# )

# mlp_label_encoder = LabelEncoder()
# y_train_mlp = mlp_label_encoder.fit_transform(y_train)

# # selector_mlp = SelectKBest(chi2, k=8000)
# # X_train_small_mlp = selector_mlp.fit_transform(X_train_encoded, y_train)
# # X_test_small_mlp = selector_mlp.transform(X_test_encoded)

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

# print("\n" + "="*80)
# print("TRYING: XGBoost Classifier")
# print("="*80)

# # 97,52%

# selector_xgb = SelectKBest(chi2, k=8000)
# X_train_small_xgb = selector_xgb.fit_transform(X_train_encoded, y_train)
# X_test_small_xgb = selector_xgb.transform(X_test_encoded)

# xgb_label_encoder = LabelEncoder()
# y_train_xgb_full = xgb_label_encoder.fit_transform(y_train)
# y_test_xgb = xgb_label_encoder.transform(y_test)

# X_train_xgb, X_val_xgb, y_train_xgb, y_val_xgb = train_test_split(
#     X_train_small_xgb,
#     y_train_xgb_full,
#     test_size=0.1,
#     random_state=42
# )

# xgb_clf = xgb.XGBClassifier(
#     objective='multi:softmax',
#     num_class=len(xgb_label_encoder.classes_),
#     n_estimators=300,
#     max_depth=10,
#     # learning_rate=0.2,
#     early_stopping_rounds=20,
#     tree_method='hist',
#     random_state=0,
#     n_jobs=-1,
#     eval_metric='mlogloss'
# )

# xgb_clf.fit(
#     X_train_xgb,
#     y_train_xgb,
#     eval_set=[(X_val_xgb, y_val_xgb)],
#     verbose=30
# )

# xgb_train_pred_int = xgb_clf.predict(X_train_small_xgb)
# xgb_test_pred_int = xgb_clf.predict(X_test_small_xgb)

# xgb_train_pred = xgb_label_encoder.inverse_transform(xgb_train_pred_int)
# xgb_test_pred = xgb_label_encoder.inverse_transform(xgb_test_pred_int)

# evaluate_model("XGBoost Classifier", y_test, xgb_test_pred, xgb_train_pred)


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

# print("\n" + "="*80)
# print("TRYING: Perceptron")
# print("="*80)

# # 96,04%

# perc = Perceptron(
#     penalty=None,
#     alpha=1e-3,
#     l1_ratio=0.15,
#     fit_intercept=True,
#     max_iter=200,
#     shuffle=True,
#     verbose=0,
#     n_jobs=-1,
#     random_state=0,
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=20,
#     class_weight='balanced'
# )

# perc.fit(X_train_encoded, y_train)
# perc_train_pred = perc.predict(X_train_encoded)
# perc_test_pred = perc.predict(X_test_encoded)
# evaluate_model("Perceptron", y_test, perc_test_pred, perc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Perceptron With Manual Weights")
# print("="*80)

# # 96,28%

# class_weight_map = {
#     'Necessary': 3.0,      
#     'Preferences': 2.5,
#     'Statistics': 1.3,
#     'Marketing': 1.0,    
# } 

# perc_mw = Perceptron(
#     penalty=None,
#     alpha=1e-3,
#     l1_ratio=0.15,
#     fit_intercept=True,
#     max_iter=100,
#     shuffle=True,
#     verbose=0,
#     n_jobs=-1,
#     random_state=0,
#     early_stopping=True,
#     validation_fraction=0.1,
#     n_iter_no_change=10,
#     class_weight=class_weight_map
# )

# perc_mw.fit(X_train_encoded, y_train)
# perc_mw_train_pred = perc_mw.predict(X_train_encoded)
# perc_mw_test_pred = perc_mw.predict(X_test_encoded)
# evaluate_model("Perceptron With Manual Weights", y_test, perc_mw_test_pred, perc_mw_train_pred)

# print("\n" + "="*80)
# print("TRYING: Nearest Centroid")
# print("="*80)

# # 78,14%

# selector_bnb = SelectKBest(chi2, k=1500) 
# X_train_small_nc = selector_bnb.fit_transform(X_train_encoded, y_train)
# X_test_small_nc = selector_bnb.transform(X_test_encoded)


# nc = NearestCentroid(metric='euclidean', shrink_threshold=0.001, priors='uniform')

# nc.fit(X_train_small_nc, y_train)
# nc_train_pred = nc.predict(X_train_small_nc)
# nc_test_pred = nc.predict(X_test_small_nc)
# evaluate_model("Nearest Centroind", y_test, nc_test_pred, nc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Ridge Classifier")
# print("="*80)

# # 95,24%

# rc = RidgeClassifier(
#     alpha=0.1,
#     class_weight='balanced',
#     solver='auto',
#     random_state=0
# )

# rc.fit(X_train_encoded, y_train)
# rc_train_pred = rc.predict(X_train_encoded)
# rc_test_pred = rc.predict(X_test_encoded)
# evaluate_model("Ridge CLassifier", y_test, rc_test_pred, rc_train_pred)

# print("\n" + "="*80)
# print("TRYING: Ridge Classifier With Manual Weights")
# print("="*80)

# # 96,41% 

# class_weight_map = {
#     'Necessary': 3.0,      
#     'Preferences': 1.8,
#     'Statistics': 1.5,
#     'Marketing': 1.0,    
# } 

# rc_mw = RidgeClassifier(
#     alpha=0.1,
#     class_weight=class_weight_map,
#     solver='auto',
#     random_state=0
# )

# rc_mw.fit(X_train_encoded, y_train)
# rc_mw_train_pred = rc_mw.predict(X_train_encoded)
# rc_mw_test_pred = rc_mw.predict(X_test_encoded)
# evaluate_model("Ridge CLassifier With Manual Weights", y_test, rc_mw_test_pred, rc_mw_train_pred)

# print("\n" + "="*80)
# print("TRYING: Ridge Classifier CV")
# print("="*80)

# # 95,00%

# selector_rccv = SelectKBest(chi2, k=20000) 
# X_train_small_rccv = selector_rccv.fit_transform(X_train_encoded, y_train)
# X_test_small_rccv = selector_rccv.transform(X_test_encoded)

# cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# rccv = RidgeClassifierCV(
#     alphas=(0.1, 1.0, 10.0),
#     scoring='top_k_accuracy', # None, top_k_accuracy, average_precision, accuracy
#     cv=cv_splitter,
#     class_weight='balanced'
# )

# rccv.fit(X_train_small_rccv, y_train)
# rccv_train_pred = rccv.predict(X_train_small_rccv)
# rccv_test_pred = rccv.predict(X_test_small_rccv)
# evaluate_model("Ridge Clasifier CV", y_test, rccv_test_pred, rccv_train_pred)

# print("\n" + "="*80)
# print("TRYING: Ridge Classifier CV With Manual Weights")
# print("="*80)

# # 96,26%

# class_weight_map = {
#     'Necessary': 3.0,      
#     'Preferences': 1.5,
#     'Statistics': 1.3,
#     'Marketing': 1.0,    
# } 

# selector_rccv_mw = SelectKBest(chi2, k=15000) 
# X_train_small_rccv_mw = selector_rccv_mw.fit_transform(X_train_encoded, y_train)
# X_test_small_rccv_mw = selector_rccv_mw.transform(X_test_encoded)

# cv_splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

# rccv_mw = RidgeClassifierCV(
#     alphas=(0.1, 1.0, 10.0),
#     scoring='top_k_accuracy',
#     cv=cv_splitter,
#     class_weight=class_weight_map
# )

# rccv_mw.fit(X_train_small_rccv_mw, y_train)
# rccv_mw_train_pred = rccv_mw.predict(X_train_small_rccv_mw)
# rccv_mw_test_pred = rccv_mw.predict(X_test_small_rccv_mw)
# evaluate_model("Ridge Clasifier CV With Manual Weights", y_test, rccv_mw_test_pred, rccv_mw_train_pred)

# print("\n" + "="*80)
# print("TRYING: Calibrated Classifier CV")
# print("="*80)

# # 97,26%

# cv_splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

# cccv = CalibratedClassifierCV(method='isotonic', cv=cv_splitter, n_jobs=-1)

# cccv.fit(X_train_encoded, y_train)
# cccv_train_pred = cccv.predict(X_train_encoded)
# cccv_test_pred = cccv.predict(X_test_encoded)
# evaluate_model("Calibrated Classifier CV", y_test, cccv_test_pred, cccv_train_pred)