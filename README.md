# AI COOKIE CLASSIFIER

> Cookie classification with TF-IDF + linear models, focused on strong per-class performance under class imbalance.

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Project-Active-brightgreen.svg)

## Table of Contents

- [Overview](#overview)
- [Training data](#training-data)
- [Preprocessing](#preprocessing)
- [Model overview](#model-overview)
- [Machine Learning](#machine-learning)
	- [Pipeline](#pipeline)
	- [Models](#models)
	- [Class imbalance handling](#class-imbalance-handling)
	- [Evaluation metrics](#evaluation-metrics)
	- [Confusion matrices (PDF)](#confusion-matrices-pdf)
	- [Confusion matrices (preview images)](#confusion-matrices-preview-images)
	- [How to read the confusion matrix values](#how-to-read-the-confusion-matrix-values)

## OVERVIEW

This project aims to classify cookies using different approaches:

- Logistic Regression (balanced weights): [more](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Logistic Regression (manual weights): [more](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- LinearSVC (manual weights): [more](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html)

| Model | Weighting strategy | Strength |
|---|---|---|
| Logistic Regression | Balanced | Strong baseline for imbalanced data |
| Logistic Regression | Manual | Lets you prioritize business-critical classes |
| LinearSVC | Manual | Fast and effective for sparse text features |

---

### Training data

> Target classes: **Necessary**, **Preferences**, **Statistics**, **Marketing**

The data consists of the following fields:

- ID (just a counter)
- Cookie Name
- Type
- Provider
- Site Found
- Expiration Date
- Description

ID is just a counter for performing operations on the database; it is not used for training or testing.

Cookie Name is the given name of the cookie, in lowercase.

Type is the cookie type based on the purpose as mentioned in the GDPR classification:

- Strictly necessary
- Preferences
- Statistics
- Marketing

Here is the [link](https://gdpr.eu/cookies/) for the GDPR classification.

Provider is the domain from which the cookie came.

Site Found is the domain where the cookie is found. We remove dots and commas from the domain in order to separate words for better learning (if it comes from a different domain than the one where the cookie was found, this may suggest it is more likely to be a third-party cookie).

Expiration Date indicates how long the cookie lasts (for example: a month, a year, or a session).

Description is the cookie description as provided by us. Not all cookies have a description, and while in this case the test data can have one, real test data often does not, so we removed it since it was not improving accuracy and was actually decreasing it by a very small margin.

### Preprocessing

| Step | Purpose |
|---|---|
| Domain cleaning | Improves token quality from provider/site strings |
| Feature combination | Gives model richer context for each cookie |
| TF-IDF encoding | Converts text into weighted sparse vectors |

```mermaid
flowchart TD
	A[Raw CSV Rows] --> B[Drop ID Column]
	B --> C[Filter Valid Categories]
	C --> D[Split Train/Test 90-10]
	D --> E[Clean Provider and Site Found]
	E --> F[Lowercase Cookie_Name and Duration]
	F --> G[Combine Text Features]
	G --> H[TF-IDF Vectorization]
	H --> I[Encoded Feature Matrix]
```

We clean the domains (both Provider and Site Found) by removing dots and commas from the domain in order to separate words for better learning, and then transform them to lowercase (this transformation is also applied to Cookie Name and Duration). We combine features, and we also add Cookie Name two times to give it more weight, since it is the most important feature for classification.

We encode the data using TF-IDF with 80,000 max features (to limit memory), `ngram_range=(1, 3)` in order to include words up to trigrams, `min_df=5` to avoid junk while still keeping rare cookie patterns, and `sublinear_tf=True` to replace raw term frequency with `1 + log(tf)` so very frequent words do not dominate the model.

### Model overview

## Machine Learning

### Pipeline

The classification workflow is:

1. Load and clean cookie records.
2. Filter categories to the four target classes (`Necessary`, `Preferences`, `Statistics`, `Marketing`).
3. Split data into training and testing sets (90/10).
4. Build text features with TF-IDF from the combined cookie fields.
5. Train weighted classifiers to handle class imbalance.
6. Evaluate with confusion matrix and per-class accuracy.

### ML architecture (class diagram)

```mermaid
classDiagram
	Dataset "1" --> "1" Preprocessor : cleaned by
	Preprocessor "1" --> "1" FeatureBuilder : combines fields
	FeatureBuilder "1" --> "1" TfidfEncoder : vectorized by
	TfidfEncoder "1" --> "many" Classifier : input features
	Classifier "1" --> "1" Evaluator : predictions
	Evaluator "1" --> "many" ConfusionMatrix : generates

	class Dataset {
		+load_csv(path)
		+filter_categories()
		+split_train_test()
	}
	class Preprocessor {
		+clean_domain(text)
		+lowercase_fields()
	}
	class FeatureBuilder {
		+combine_features(df)
	}
	class TfidfEncoder {
		+max_features: 80000
		+ngram_range: (1,3)
		+min_df: 5
		+sublinear_tf: true
		+fit_transform()
		+transform()
	}
	class Classifier {
		+fit(X, y)
		+predict(X)
	}
	class Evaluator {
		+accuracy_score()
		+confusion_matrix()
		+save_pdf_png()
	}
	class ConfusionMatrix {
		+rows: true classes
		+columns: predicted classes
	}
```

### Full ML pipeline (train/evaluate/save)

```mermaid
sequenceDiagram
	participant Data as Dataset
	participant Prep as Preprocessor
	participant Vec as TF-IDF Encoder
	participant Model as Classifier
	participant Eval as Evaluator
	participant Out as Exporter

	Data->>Prep: Load, clean, filter classes
	Prep->>Data: Train/Test split
	Prep->>Vec: Combine and normalize text
	Vec->>Model: X_train, y_train
	Model->>Eval: Predictions on test set
	Eval->>Eval: Accuracy + per-class metrics
	Eval->>Out: Confusion matrix
	Out->>Out: Save PDF
	Out->>Out: Save PNG
```

### Models

The project compares multiple linear models:

- KNN
- KNN With Inver Class Weighting
- Logistic Regression
- Logistic Regression With Manual Weights
- SGD Classifier
- Passive Aggressive Classifier
- LinearSVC
- LinearSVC With Manual Weights
- Random Forest Classifier
- Decision Tree Classifier With Manual Weights
- Ada Boost Classifier
- Multinomial Naive Bayes
- Complement Naive Bayes
- Bernoulli Naive Bayes
- MLP (Multi-layer Perceptron) Classifier
- LightGBM Classifier
- XGBoost Classifier
- CatBoost Classifier
- Nearest Centroid
- Perceptron
- Perceptron with manual class weights
- Ridge Classifier
- Ridge CLassifier With Manual Weights
- Ridge Classifier CV
- Ridge Classifier CV With Manual Weights
- Stacking Classifier
- Voting Classifier
- Calibrated Classifier CV

Linear models work well for high-dimensional sparse TF-IDF features and scale better than many non-linear models on this dataset size.

### Class imbalance handling

The dataset is imbalanced, especially for `Preferences`, so weighted training is used.

- Manual weighting allows emphasizing business-critical classes (for example, `Necessary`).
- Balanced weighting uses class frequency automatically.

This improves minority-class recall while preserving high overall accuracy.

### Evaluation metrics

For each model, the script reports:

- Training accuracy
- Test accuracy
- Confusion matrix
- Per-category accuracy

The confusion matrix is also saved as a PDF file for each model run.

### Confusion matrices (PDF)

You can find the generated confusion matrices here:

- [KNN](confusion_matrix_KNN.pdf)
- [KNN With Inverse Class Weighting](confusion_matrix_KNN_WITH_INVERSE_CLASS_WEIGHTING.pdf)
- [Logistic Regression](confusion_matrix_Logistic_Regression_Balanced.pdf)
- [Logistic Regression With Manual Weights)](confusion_matrix_Logistic_Regression_Manual_Weights.pdf)
- [SGD Classifier](confusion_matrix_SGD_Classifier.pdf)
- [Passive Aggressive Classifier](confusion_matrix_Passive_Aggressive_Classifier.pdf)
- [LinearSVC](confusion_matrix_LinearSVC.pdf)
- [LinearSVC Manual Weights](confusion_matrix_LinearSVC_Manual_Weights.pdf)
- [Random Forest Classifier](confusion_matrix_RandomForest_CLassifier.pdf)
- [Decision Tree Classifier With Manual Weights](confusion_matrix_Decision_Tree_Classifier.pdf)
- [Ada Boost Classifier](confusion_matrix_Ada_Boost_Classifier.pdf)
- [Multinomial Naive Bayes](confusion_matrix_Multinomial_Naive_Bayes.pdf)
- [Complement Naive Bayes](confusion_matrix_Complement_Naive_Bayes.pdf)
- [Bernoulli Naive Bayes](confusion_matrix_Bernoulli_Naive_Bayes.pdf)
- [MLP (Multi-layer Perceptron) Classifier](confusion_matrix_MLP_Classifier.pdf)
- [LightGBM Classifier](confusion_matrix_LightGBM_Classifier.pdf)
- [XGBoost Classifier](confusion_matrix_XGBoost_Classifier.pdf)
- [CatBoost Classifier](confusion_matrix_CatBoost_Classifier.pdf)
- [Nearest Centroid](confusion_matrix_Nearest_Centroind.pdf)
- [Perceptron](confusion_matrix_Perceptron.pdf)
- [Perceptron With Manual Weights](confusion_matrix_Perceptron_With_Manual_Weights.pdf)
- [Ridge CLassifier](confusion_matrix_Ridge_CLassifier.pdf)
- [Ridge CLassifier With Manual Weights](confusion_matrix_Ridge_CLassifier_With_Manual_Weights.pdf)
- [Ridge Classifier CV](confusion_matrix_Ridge_Clasifier_CV.pdf)
- [Ridge Classifier CV With Manual Weights](confusion_matrix_Ridge_Classifier_CV_With_Manual_Weights.pdf)
- [Stacking Classifier](confusion_matrix_Stacking_Classifier.pdf)
- [Voting Classifier](confusion_matrix_Voting_Classifier.pdf)
- [Calibrated Classifier CV](confusion_matrix_Calibrated_Classifier_CV.pdf)

### Confusion matrices (preview images)

#### KNN

![KNN](confusion_matrix_KNN.png)

#### KNN With Inverse Class Weighting

![KNN With Inverse Class Weighting](confusion_matrix_KNN_WITH_INVERSE_CLASS_WEIGHTING.png)

#### Logistic Regression (Manual Weights)

![Logistic Regression (Manual Weights)](confusion_matrix_Logistic_Regression_Manual_Weights.png)

#### Logistic Regression (Balanced)

![Logistic Regression (Balanced)](confusion_matrix_Logistic_Regression_Balanced.png)

#### SGD Classifier

![SGD Classifier](confusion_matrix_SGD_Classifier.png)

#### Passive Aggressive Classifier

![Passive Aggressive Classifier](confusion_matrix_Passive_Aggressive_Classifier.png)

#### LinearSVC

![LinearSVC](confusion_matrix_LinearSVC.png)

#### LinearSVC Manual Weights

![LinearSVC Manual Weights](confusion_matrix_LinearSVC_Manual_Weights.png)

#### Random Forest Classifier

![Random Forest Classifier](confusion_matrix_RandomForest_Classifier.png))

#### Decision Tree Classifier (Manual Weights)

![Decision Tree Classifier (Manual Weights)](confusion_matrix_Decision_Tree_Classifier.png)

#### Ada Boost Classifier

![Ada Boost Classifier](confusion_matrix_Ada_Boost_Classifier.png)

#### Multinomial Naive Bayes

![Multinomial Naive Bayes](confusion_matrix_Multinomial_Naive_Bayes.png)

#### Complement Naive Bayes

![Complement Naive Bayes](confusion_matrix_Complement_Naive_Bayes.png)

#### Bernoulli Naive Bayes

![Bernoulli Naive Bayes](confusion_matrix_Bernoulli_Naive_Bayes.png)

#### MLP (Multi-layer Perceptron) Classifier

![MLP (Multi-layer Perceptron) Classifier](confusion_matrix_MLP_Classifier.png)

#### LightGBM Classifier
![LightGBM Classifier](confusion_matrix_LightGBM_Classifier.png)

#### XGBoost Classifier

![XGBoost Classifier](confusion_matrix_XGBoost_Classifier.png)

#### CatBoost CLassifier

![CatBoost Classifier](confusion_matrix_CatBoost_Classifier.png)

#### Nearest Centroid

![Nearest Centroid](confusion_matrix_Nearest_Centroind.png)

#### Perceptron

![Perceptron](confusion_matrix_Perceptron.png)

#### Perceptron With Manual Weights

![Perceptron With Manual Weights](confusion_matrix_Perceptron_With_Manual_Weights.png)

#### Ridge Classifier

![Ridge Classifier](confusion_matrix_Ridge_Classifier.png)

#### Ridge CLassifier With Manual Weights

![Ridge CLassifier With Manual Weights](confusion_matrix_Ridge_CLassifier_With_Manual_Weights.png)

#### Ridge Classifier CV

![Ridge Classifier CV](confusion_matrix_Ridge_Clasifier_CV.png)

#### Ridge Classifier CV With Manual Weights

![Ridge Classifier CV With Manual Weights](confusion_matrix_Ridge_Classifier_CV_With_Manual_Weights.png)

#### Stacking Classifier

![Stacking Classifier](confusion_matrix_Stacking_Classifier.png)

#### Voting Classifier

![Voting Classifier](confusion_matrix_Voting_Classifier.png)

#### Calibrated Classifier CV

![Calibrated Classifier CV](confusion_matrix_Calibrated_Classifier_CV.pdf)

### How to read the confusion matrix values

Each row is the **true class** and each column is the **predicted class**.

- A value on the diagonal (top-left to bottom-right) means a correct prediction.
- A value off the diagonal means a misclassification.
- Larger diagonal values are better.

For example, if the row is `Preferences` and the column is `Marketing`, that value is the number of `Preferences` cookies incorrectly predicted as `Marketing`.

#### Notes for this project

- `Necessary` is business-critical, so we monitor its row carefully to reduce false negatives.
- `Preferences` is the smallest class, so confusion with `Marketing` or `Statistics` can happen more often.
- A model can have high overall accuracy while still underperforming on minority classes, so per-class values are always checked.
