import pandas as pd, matplotlib.pyplot as plt, seaborn as sns, numpy as np, re, json, lightgbm as lgb, xgboost as xgb, catboost, gc, joblib
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, normalize
from scipy.sparse import hstack

from skl2onnx import convert_sklearn 
from skl2onnx.common.data_types import FloatTensorType

CONFUSION_MATRICES_PDF_DIR = Path('confusion_matrices_pdf')
CONFUSION_MATRICES_PNG_DIR = Path('confusion_matrices_png')
DATA_ANALYSIS_PDF_DIR = Path('training_data_analysis_pdf')
DATA_ANALYSIS_PNG_DIR = Path('training_data_analysis_png')
MODEL_ACCURACY_DATA_DIR = Path('model_accuracies')

# hasing pe date

class CookieDataLoader:
    def __init__(self, csv_path='updated_classified_cookies.csv', test_size=0.1, random_state=42):
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
        # train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state, stratify=df["Category"],)
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
        
        def parse_duration(text):
            if pd.isna(text) or text.strip() == '':
                return ''
            
            text = text.lower().strip()
            
            # session case
            if 'session' in text:
                return 'session'
            
            if 'never' in text:
                return 'long'
            
            # extract numbers
            years = months = days = minutes = 0
            
            year_match = re.search(r'(\d+)\s*year', text)
            month_match = re.search(r'(\d+)\s*month', text)
            day_match = re.search(r'(\d+)\s*day', text)
            minute_match = re.search(r'(\d+)\s*minute', text)
            
            if year_match:
                years = int(year_match.group(1))
            if month_match:
                months = int(month_match.group(1))
            if day_match:
                days = int(day_match.group(1))
            if minute_match:
                minutes = int(minute_match.group(1))
            
            total_days = years * 365 + months * 30 + days + minutes / (60 * 24)
            
            if total_days > 1:
                return 'long'
            elif total_days <= 1:
                return 'short'
            else:
                return ''
            
        return (df['Cookie_Name'].fillna('')  + ' ' +
                df['Provider'].apply(clean_domain) + ' ' +
                df['Site_Found'].apply(clean_domain) + ' ' +
                df['Duration'].apply(parse_duration))
    
    def build_tfidf(self, max_features=80000):
        self.X_train_text = self.combine_features(self.X_train_raw)
        self.X_test_text = self.combine_features(self.X_test_raw)

        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1,2),
            min_df=10,
            sublinear_tf=True
        )

        self.X_train_vec = self.tfidf.fit_transform(self.X_train_text).astype(np.float32)
        self.X_test_vec = self.tfidf.transform(self.X_test_text).astype(np.float32)
    
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
        # Train/Test data vectorized to reuse
        self.X_train_vec = loader.X_train_vec
        self.X_test_vec = loader.X_test_vec
        self.model_name = model_name
        self.use_custom_weights = use_custom_weights
        self.max_features = None if max_features is True else max_features
        self.k_best = k_best
        self.mlp_label_encoder = None
        # Handle char max_features: True -> None (all char features), False -? skip
        self.max_features_chars = None if max_features_chars is True else max_features_chars
        self.k_best_chars = k_best_chars
        # Prepare text and labels
        self._prepare_text()
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
                sublinear_tf=True
            )
            self.X_train_chars = self.tfidf_chars.fit_transform(self.X_train_text)
            self.X_test_chars = self.tfidf_chars.transform(self.X_test_text)

            self.X_train_chars = self.X_train_chars.astype("float32")
            self.X_test_chars = self.X_test_chars.astype("float32")

           

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
        if self.model_name == "MLP": # 98.81%
            self.model = MLPClassifier(
                    hidden_layer_sizes=(250, 64),   
                    activation='relu', # tanh has 98.70%
                    solver='adam',              
                    alpha=1e-3,                 
                    batch_size=8192,
                    learning_rate_init=0.01,
                    max_iter=200,
                    verbose=True,
                    early_stopping=True,
                    validation_fraction=0.01,
                    beta_1=0.9,
                    epsilon=1e-8,    
                    n_iter_no_change=10,
                    random_state=42,
            )
        elif self.model_name == "KNN": # 95.18%
            def KNN_custom_weights(distances):
                return (1 / (distances + 1e-5) ** 2.2) * np.exp(-distances)
            
            self.model = KNeighborsClassifier(
                n_neighbors=7,            
                weights=KNN_custom_weights,
                metric='euclidean',     
                algorithm='auto',
                n_jobs=-1
            )
        elif self.model_name == "LogisticRegression": # 93.72%
            self.model = LogisticRegression(
                    max_iter=10000, 
                    solver='saga', 
                    random_state=0, 
                    class_weight='balanced'
            )
        elif self.model_name == "LogisticRegression_MW": # 95.52%
            self.model = LogisticRegression(
                    C=1.0,
                    max_iter=10000,
                    solver='saga',  
                    random_state=0
            )
        elif self.model_name == "SGD": # 97.62%
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
        elif self.model_name == "PAC": # 97.35%
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
        elif self.model_name == "LinearSVC": # 97.22%
            self.model = LinearSVC(
                C=1.3,
                class_weight='balanced',
                max_iter=20000,
                random_state=0,
                penalty='l1'
            )
        elif self.model_name == "LinearSVC_MW": # 97.37%
            self.model = LinearSVC(  
                C=0.6,
                max_iter=20000,
                random_state=0,
                penalty='l1',
                class_weight={'Necessary': 3.5, 'Preferences': 2.7, 'Statistics': 1.7, 'Marketing': 1.0},
            )
        elif self.model_name == "RandomForest": # 98.18%
            self.model = RandomForestClassifier(
                n_estimators=300,
                random_state=0,
                n_jobs=-1,
                class_weight='balanced_subsample',
                max_depth=None,
                max_features='sqrt'   
            )
        elif self.model_name == "DecisionTreeClassifier": # 98.17%
            self.model = DecisionTreeClassifier( 
                criterion='gini',
                max_depth=None,
                max_features=None,
                class_weight={"Necessary": 3.5, "Preferences": 2.7, "Statistics": 1.7, "Marketing": 1.0},
                random_state=0
            )
        elif self.model_name == "AdaBoost": # 94.23%
            ada_base_tree = DecisionTreeClassifier(
                criterion='gini',
                max_depth=15,
                random_state=0
            )

            self.model = AdaBoostClassifier(
                estimator=ada_base_tree,
                n_estimators=200,
                learning_rate=0.4,
                random_state=0
            )
        elif self.model_name == "MultinomialNB": # 94.27%
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

        elif self.model_name == "ComplementNB": # 94.87%
            self.word_vec_cnb = TfidfVectorizer(ngram_range=(1,2), min_df=5, sublinear_tf=True, dtype=np.float32)

            Xtr_w = self.word_vec_cnb.fit_transform(self.X_train_text)
            Xte_w = self.word_vec_cnb.transform(self.X_test_text)

            self.char_vec_cnb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6), min_df=9, sublinear_tf=True, dtype=np.float32)

            Xtr_c = self.char_vec_cnb.fit_transform(self.X_train_text)
            Xte_c = self.char_vec_cnb.transform(self.X_test_text)

            k_word = 50000
            k_char = 35000

            self.sel_w_cnb = SelectKBest(chi2, k=k_word)
            self.sel_c_cnb = SelectKBest(chi2, k=k_char)

            Xtr_w_sel = self.sel_w_cnb.fit_transform(Xtr_w, self.y_train)
            Xte_w_sel = self.sel_w_cnb.transform(Xte_w)

            Xtr_c_sel = self.sel_c_cnb.fit_transform(Xtr_c, self.y_train)
            Xte_c_sel = self.sel_c_cnb.transform(Xte_c)

            # Combine features
            self.X_train_vec = hstack([Xtr_w_sel, Xtr_c_sel])
            self.X_test_vec = hstack([Xte_w_sel, Xte_c_sel])

            self.model = ComplementNB(alpha=1e-8)

            self.sample_weights = compute_sample_weight(
                class_weight={"Necessary": 1.8, "Preferences": 1.4, "Statistics": 1.3, "Marketing": 1.0},
                y=self.y_train
            )

        elif self.model_name == "BernoulliNB": # 93,75%
            # Word-level TF-IDFs
            self.word_vec_bnb = TfidfVectorizer(ngram_range=(1,2), min_df=10, sublinear_tf=True, dtype=np.float32)

            Xtr = self.word_vec_bnb.fit_transform(self.X_train_text)
            Xte = self.word_vec_bnb.transform(self.X_test_text)

            # Convert to binary features
            Xtr_bin = (Xtr > 0).astype(np.int8)
            Xte_bin = (Xte > 0).astype(np.int8)

            # Feature selection
            k = 22000
            self.sel_bnb = SelectKBest(chi2, k=k)

            Xtr_sel = self.sel_bnb.fit_transform(Xtr_bin, self.y_train)
            Xte_sel = self.sel_bnb.transform(Xte_bin)

            self.X_train_vec = Xtr_sel
            self.X_test_vec = Xte_sel

            self.model = BernoulliNB(alpha=1e-3, binarize=None)

        elif self.model_name == "StackingClassifier": # 95.76

            base_estimators = [
                ("lr", LogisticRegression(
                    max_iter=3000,
                    solver="saga",
                    class_weight="balanced",
                    random_state=0
                )),
                ("sgd", SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=0
                )),
                ("cnb", ComplementNB(alpha=0.1))
            ]

            meta_model = LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=0
            )

            self.model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_model,
                cv=3,
                n_jobs=-1,
                passthrough=False
            )

        elif self.model_name == "VotingClassifier": # 96.40%

            self.selector_voting = SelectKBest(chi2, k=10000)

            Xtr = self.selector_voting.fit_transform(self.loader.X_train_vec, self.y_train)
            Xte = self.selector_voting.transform(self.loader.X_test_vec)

            self.X_train_vec = Xtr
            self.X_test_vec = Xte

            self.model = VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(
                        max_iter=5000,
                        solver="saga",
                        class_weight="balanced",
                        random_state=0
                    )),
                    ("sgd", SGDClassifier(
                        loss="log_loss",
                        alpha=1e-7,
                        max_iter=100,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                        class_weight={0: 3.0, 1: 2.5, 2: 1.0, 3: 1.0},
                        random_state=0
                    )),
                    ("cnb", ComplementNB(alpha=1e-8))
                ],
                voting="hard",
                n_jobs=-1,
                verbose=10
            )

        elif self.model_name == "LightGBM": # 98.47%

            self.model = lgb.LGBMClassifier(
                num_iterations=700,
                num_leaves=120,
                random_state=0,
                n_jobs=-1,
                force_col_wise=True
            )

        elif self.model_name == "XGBoost": # 97.52%

            self.selector_xgb = SelectKBest(chi2, k=8000)

            Xtr = self.selector_xgb.fit_transform(self.X_train_vec, self.y_train)
            Xte = self.selector_xgb.transform(self.X_test_vec)

            self.X_train_vec = Xtr
            self.X_test_vec = Xte

            self.xgb_label_encoder = LabelEncoder()

            y_train_encoded = self.xgb_label_encoder.fit_transform(self.y_train)
            self.y_test_encoded = self.xgb_label_encoder.transform(self.y_test)

            self.X_train_xgb, self.X_val_xgb, self.y_train_xgb, self.y_val_xgb = train_test_split(
                self.X_train_vec,
                y_train_encoded,
                test_size=0.1,
                random_state=42
            )

            self.model = xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=len(self.xgb_label_encoder.classes_),
                n_estimators=300,
                max_depth=10,
                early_stopping_rounds=20,
                tree_method='hist',
                random_state=0,
                n_jobs=-1,
                eval_metric='mlogloss'
            )

        elif self.model_name == "CatBoost": # 97.63%

            self.selector_cat = SelectKBest(chi2, k=8000)

            Xtr = self.selector_cat.fit_transform(self.X_train_vec, self.y_train)
            Xte = self.selector_cat.transform(self.X_test_vec)

            self.X_train_vec = Xtr
            self.X_test_vec = Xte

            self.X_train_cat, self.X_val_cat, self.y_train_cat, self.y_val_cat = train_test_split(
                self.X_train_vec,
                self.y_train,
                test_size=0.1,
                random_state=42
            )

            class_weights = {
                'Necessary': 3.0,
                'Preferences': 2.5,
                'Statistics': 1.3,
                'Marketing': 1.0
            }

            self.model = catboost.CatBoostClassifier(
                iterations=2000,
                learning_rate=0.7,
                depth=6,
                loss_function='MultiClass',
                eval_metric='Accuracy',
                class_weights=class_weights,
                random_seed=42,
                od_type='IncToDec',
                od_wait=20,
                use_best_model=True,
                task_type='CPU',
                thread_count=-1,
                verbose=50
            )

        elif self.model_name == "Perceptron": # 96.04%

            self.model = Perceptron(
                penalty=None,
                alpha=1e-3,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=200,
                shuffle=True,
                verbose=0,
                n_jobs=-1,
                random_state=0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
                class_weight='balanced'
            )

        elif self.model_name == "Perceptron_MW": # 96.28%

            class_weight_map = {
                'Necessary': 3.0,
                'Preferences': 2.5,
                'Statistics': 1.3,
                'Marketing': 1.0
            }

            self.model = Perceptron(
                penalty=None,
                alpha=1e-3,
                l1_ratio=0.15,
                fit_intercept=True,
                max_iter=100,
                shuffle=True,
                verbose=0,
                n_jobs=-1,
                random_state=0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                class_weight=class_weight_map
            )

        elif self.model_name == "NearestCentroid": # 90.17% 
            self.selector_nc = SelectKBest(chi2, k=2000) 
            Xtr = self.selector_nc.fit_transform(self.X_train_vec, self.y_train) 
            Xte = self.selector_nc.transform(self.X_test_vec)

            Xtr = Xtr.astype(np.float32).toarray()
            Xte = Xte.astype(np.float32).toarray()

            self.X_train_vec = Xtr 
            self.X_test_vec = Xte 

            self.model = NearestCentroid( 
                    metric='euclidean', 
                    shrink_threshold=0.4,
                    priors='empirical' 
                ) 

        elif self.model_name == "RidgeClassifier": # 95.24%

            self.model = RidgeClassifier(
                alpha=0.1,
                class_weight='balanced',
                solver='auto',
                random_state=0
            )

        elif self.model_name == "RidgeClassifier_MW": # 96.41%

            class_weight_map = {
                'Necessary': 3.0,
                'Preferences': 1.8,
                'Statistics': 1.5,
                'Marketing': 1.0
            }

            self.model = RidgeClassifier(
                alpha=0.1,
                class_weight=class_weight_map,
                solver='auto',
                random_state=0
            )

        elif self.model_name == "RidgeClassifierCV": # 95.00%

            self.selector_rccv = SelectKBest(chi2, k=20000)

            Xtr = self.selector_rccv.fit_transform(self.X_train_vec, self.y_train)
            Xte = self.selector_rccv.transform(self.X_test_vec)

            self.X_train_vec = Xtr
            self.X_test_vec = Xte

            cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            self.model = RidgeClassifierCV(
                alphas=(0.1, 1.0, 10.0),
                scoring='top_k_accuracy',
                cv=cv_splitter,
                class_weight='balanced'
            )

        elif self.model_name == "RidgeClassifierCV_MW": # 96.26%

            class_weight_map = {
                'Necessary': 3.0,
                'Preferences': 1.5,
                'Statistics': 1.3,
                'Marketing': 1.0
            }

            self.selector_rccv_mw = SelectKBest(chi2, k=15000)

            Xtr = self.selector_rccv_mw.fit_transform(self.X_train_vec, self.y_train)
            Xte = self.selector_rccv_mw.transform(self.X_test_vec)

            self.X_train_vec = Xtr
            self.X_test_vec = Xte

            cv_splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

            self.model = RidgeClassifierCV(
                alphas=(0.1, 1.0, 10.0),
                scoring='top_k_accuracy',
                cv=cv_splitter,
                class_weight=class_weight_map
            )

        elif self.model_name == "CalibratedClassifierCV": # 97.26%

            cv_splitter = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

            self.model = CalibratedClassifierCV(
                method='isotonic',
                cv=cv_splitter,
                n_jobs=-1
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
        
        self.svd_ada = TruncatedSVD(n_components=300, random_state=42)

        self.X_train_vec = self.svd_ada.fit_transform(self.X_train_vec)
        self.X_test_vec = self.svd_ada.transform(self.X_test_vec)

        self.class_weights_ada = {'Necessary': 2.0, 'Preferences': 1.5, 'Statistics': 1.0, 'Marketing': 1.0}

        subset_size = 300000 

        self.X_train_sub, self.y_train_sub = resample(
            self.X_train_vec,
            self.y_train,
            n_samples=subset_size,
            stratify=self.y_train,
            random_state=42
        )

        self.sample_weights_sub = self.y_train_sub.map(self.class_weights_ada).values

    def train(self):
        if self.model_name == "MLP": # 98.77%
            self.mlp_label_encoder = LabelEncoder()
            self.y_train_mlp = self.mlp_label_encoder.fit_transform(self.y_train)
            self.model.fit(self.X_train_vec, self.y_train_mlp)
        elif self.model_name == "KNN": # 95.18%
            self._reduce_for_knn()
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "LogisticRegression": # 93.72%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "LogisticRegression_MW": # 95.52%
            class_weight_map = {'Necessary': 5.0, 'Preferences': 2.0, 'Statistics': 1.5, 'Marketing': 1.0}      
            sample_weight = self.y_train.map(class_weight_map).fillna(1.0)
            self.model.fit(self.X_train_vec, self.y_train, sample_weight=sample_weight)
        elif self.model_name == "SGD": # 97.62%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "PAC": # 97.35%
            self.model.fit(self.X_train_vec, self.y_train) 
        elif self.model_name == "LinearSVC": # 97.22%
            self.model.fit(self.X_train_vec, self.y_train) 
        elif self.model_name == "LinearSVC_MW": # 97.37%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "RandomForest": # 98.18%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "DecisionTreeClassifier": # 98.17%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "AdaBoost": # 94.23%
            self._reduce_for_adaboost()
            self.model.fit(self.X_train_sub, self.y_train_sub, sample_weight=self.sample_weights_sub)
        elif self.model_name == "MultinomialNB": # 94.27%
            self.model.fit(self.X_train_vec, self.y_train, sample_weight=self.sample_weights)
        elif self.model_name == "ComplementNB": # 94.87%
            self.model.fit(self.X_train_vec, self.y_train, sample_weight=self.sample_weights)
        elif self.model_name == "BernoulliNB": # 93.75%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "StackingClassifier": # 95.76%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "VotingClassifier": # 96.40%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "LightGBM": # 98.47%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "XGBoost": # 97.52%
            self.model.fit(self.X_train_xgb, self.y_train_xgb, eval_set=[(self.X_val_xgb, self.y_val_xgb)], verbose=30)
        elif self.model_name == "CatBoost": # 97.63%
            self.model.fit(self.X_train_cat, self.y_train_cat, eval_set=[(self.X_val_cat, self.y_val_cat)])
        elif self.model_name == "Perceptron": # 96.04%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "Perceptron_MW": # 96.28%
            self.model.fit(self.X_train_vec, self.y_train) 
        elif self.model_name == "NearestCentroid": # 78.14%
            self.model.fit(self.X_train_vec, self.y_train) 
        elif self.model_name == "RidgeClassifier": # 95.24%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "RidgeClassifier_MW": # 96.41%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "RidgeClassifierCV": # 95.00%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "RidgeClassifierCV_MW": # 96.26%
            self.model.fit(self.X_train_vec, self.y_train)
        elif self.model_name == "CalibratedClassifierCV": # 97.26%
            self.model.fit(self.X_train_vec, self.y_train)
        else:
            self.model.fit(self.X_train_vec, self.y_train)

    def evaluate(self):
        if self.model_name == "MLP":
            train_pred_int = self.model.predict(self.X_train_vec)
            test_pred_int = self.model.predict(self.X_test_vec)
            train_pred = self.mlp_label_encoder.inverse_transform(train_pred_int)
            test_pred = self.mlp_label_encoder.inverse_transform(test_pred_int)
        elif self.model_name == "XGBoost":
            train_pred_int = self.model.predict(self.X_train_vec)
            test_pred_int = self.model.predict(self.X_test_vec)

            train_pred = self.xgb_label_encoder.inverse_transform(train_pred_int)
            test_pred = self.xgb_label_encoder.inverse_transform(test_pred_int)
        elif self.model_name == "CatBoost":
                train_pred = self.model.predict(self.X_train_vec)
                test_pred = self.model.predict(self.X_test_vec)

                # Fix: flatten
                train_pred = np.array(train_pred).ravel()
                test_pred = np.array(test_pred).ravel()
        else:
            train_pred = self.model.predict(self.X_train_vec)
            test_pred = self.model.predict(self.X_test_vec)
        train_acc = accuracy_score(self.y_train, train_pred)
        test_acc = accuracy_score(self.y_test, test_pred)
        train_f1 = f1_score(self.y_train, train_pred, average='macro', zero_division=0)
        test_f1 = f1_score(self.y_test, test_pred, average='macro', zero_division=0)
        print(f"{self.model_name} - Training Accuracy: {train_acc*100:.2f}%, Test Accuracy: {test_acc*100:.2f}%")
        print(f"{self.model_name} - Training F1: {train_f1*100:.2f}%, Test F1: {test_f1*100:.2f}%")

        # Confusion matrix (saved as PDF + PNG, same style as evaluate_model())
        classes = sorted(set(self.y_train) | set(self.y_test) | set(test_pred))
        conf_matrix = confusion_matrix(self.y_test, test_pred, labels=classes)

        print(f"\nConfusion Matrix ({self.model_name}):")
        print(f"Categories: {classes}")
        print(conf_matrix)

        print("\nPer-category accuracy:")
        per_category_accuracy = {}
        for i, cat in enumerate(classes):
            correct = int(conf_matrix[i, i])
            total = int(conf_matrix[i, :].sum())
            acc = (correct / total * 100) if total > 0 else 0
            formatted = f"{correct}/{total} = {acc:.2f}%"
            per_category_accuracy[cat] = formatted
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
        MODEL_ACCURACY_DATA_DIR.mkdir(parents=True, exist_ok=True)

        pdf_path = CONFUSION_MATRICES_PDF_DIR / f'confusion_matrix_{safe_name}.pdf'
        png_path = CONFUSION_MATRICES_PNG_DIR / f'confusion_matrix_{safe_name}.png'
        model_accuracy_path = MODEL_ACCURACY_DATA_DIR / f'{safe_name}_accuracies.json'

        plt.savefig(pdf_path, format='pdf')
        plt.savefig(png_path, format='png')
        plt.close()
        print(f"Saved confusion matrix to: {png_path} and {pdf_path}")

        results = {
            "model_name": self.model_name,
            "training_accuracy": f"{train_acc*100:.2f}%",
            "test_accuracy": f"{test_acc*100:.2f}%",
            "training_f1_macro": f"{train_f1*100:.2f}" if train_f1 is not None else None,
            "test_f1_macro": f"{test_f1*100:.2f}" if test_f1 is not None else None,
            "per_category_accuracy(correct/total)": per_category_accuracy
        }

        with open(model_accuracy_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Saved accuracy data to: {model_accuracy_path}")


data_loader = CookieDataLoader()
data_loader.build_tfidf()
CookieDataLoader.data_statistics(data_loader.df)

# CookieClassifier(
#     loader=data_loader,
#     model_name="MLP",
#     max_features=80000,
#     k_best=None,
#     max_features_chars=False,
#     k_best_chars=None
# )

classifier = CookieClassifier( # 99.15%
    loader=data_loader,
    model_name="MLP",
    max_features=80000,
    max_features_chars=20000,
)

joblib.dump({
    "tfidf_word": data_loader.tfidf,
    "tfidf_char": classifier.tfidf_chars,
    "selector_word": getattr(classifier, "selector", None),
    "selector_char": getattr(classifier, "selector_chars", None),
}, "preprocessing.joblib")

n_features = classifier.X_train_vec.shape[1]

onnx_model = convert_sklearn(
    classifier.model,
    initial_types=[("input", FloatTensorType([None, n_features]))],
    options={id(classifier.model): {"zipmap": False}},
)

with open("cookie_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

models_to_run = [
    # "MLP",                    # 98.81%
    # "KNN",                    # 95.18%
    # "LogisticRegression",     # 93.72%
    # "LogisticRegression_MW",  # 95.52%  94.14%
    # "SGD",                    # 97.62%  97.55%
    # "PAC",                    # 97.35%  97.31%
    # "LinearSVC",              # 97.22%  
    # "LinearSVC_MW",           # 97.37%
    # "RandomForest",           # 98.18%
    # "DecisionTreeClassifier", # 98.17%
    # "AdaBoost",               # 94.26%
    # "MultinomialNB",          # 94.27%
    # "ComplementNB",           # 94.87%
    # "BernoulliNB",            # 93.75%
    # "StackingClassifier",     # 95.76% 95.38%
    # "VotingClassifier",       # 96.40% 95.20%
    # "LightGBM",               # 98.48%
    # "XGBoost",                # 97.52%
    # "CatBoost",               # 97.73%
    # "Perceptron",             # 96.04% 95.94%
    # "Perceptron_MW",          # 96.28% 96.05%
    # "NearestCentroid",        # 78.14%
    # "RidgeClassifier",        # 95.25%
    # "RidgeClassifier_MW",     # 96.41%
    # "RidgeClassifierCV",      # 95.00%
    # "RidgeClassifierCV_MW",   # 96.26%
    # "CalibratedClassifierCV"  # 97.26%
]
