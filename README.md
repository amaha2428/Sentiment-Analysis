## README

### Project Title
**Sentiment Analysis Using Various Machine Learning Algorithms**

### Project Overview
This project involves building and evaluating multiple machine learning models to classify text data into sentiment labels. The dataset contains text and associated labels. The key steps include data cleaning, text vectorization using TF-IDF, training various classifiers, balancing the dataset using SMOTE, and hyperparameter tuning for RandomForestClassifier.

### Requirements
To run this project, you need the following libraries installed in your Python environment:
- pandas
- numpy
- matplotlib
- scikit-learn
- nltk
- imbalanced-learn

You can install these libraries using pip:
```bash
pip install pandas numpy matplotlib scikit-learn nltk imbalanced-learn
```

### Dataset
The project uses the following dataset:
1. **train.csv** - Contains the text data and their respective labels.

### Code Explanation

1. **Importing Libraries**
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.linear_model import LogisticRegression
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.svm import SVC
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import f1_score, accuracy_score
   from imblearn.over_sampling import SMOTE
   from sklearn.model_selection import GridSearchCV
   import nltk
   from nltk.corpus import stopwords
   from nltk.stem import WordNetLemmatizer
   import re
   from string import punctuation
   ```

2. **Loading and Exploring the Data**
   ```python
   df = pd.read_csv('train.csv')
   df.head(5)
   df['label'].value_counts().plot(kind='bar')
   plt.title('count of labels')
   plt.xlabel('Label')
   plt.ylabel('count')
   ```

3. **Cleaning the Text Data**
   ```python
   stop_words = stopwords.words("english")
   
   def cleaning_text(text, remove_stop_words=True, lemmatize_words=True):
       text = re.sub(r"[^A-Za-z0-9]", " ", text)
       text = re.sub(r"\'s", " ", text)
       text = re.sub(r"http\S+", " link ", text)
       text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)
       text = text.lower()
       text = "".join([c for c in text if c not in punctuation])
       if remove_stop_words:
           text = text.split()
           text = [w for w in text if not w in stop_words]
           text = " ".join(text)
       if lemmatize_words:
           text = text.split()
           lemmatizer = WordNetLemmatizer()
           lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
           text = " ".join(lemmatized_words)
       return text
   
   df['answer_option'] = df['answer_option'].apply(cleaning_text)
   ```

4. **Vectorizing the Text Data**
   ```python
   tf = TfidfVectorizer()
   x = tf.fit_transform(df['answer_option'])
   y = df.label
   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
   ```

5. **Training and Evaluating Models on Unbalanced Dataset**
   ```python
   ## Random Forest
   rand = RandomForestClassifier()
   rand.fit(x_train, y_train)
   rand_pred = rand.predict(x_test)
   print(f"f1 score = {f1_score(y_test, rand_pred, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test, rand_pred)*100}%")
   
   ## Logistic Regression
   log = LogisticRegression()
   log.fit(x_train, y_train)
   log_pred = log.predict(x_test)
   print(f"f1 score = {f1_score(y_test, log_pred, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test, log_pred)*100}%")
   
   ## KNeighbors Classifier
   kn = KNeighborsClassifier()
   kn.fit(x_train, y_train)
   kn_pred = kn.predict(x_test)
   print(f"f1 score = {f1_score(y_test, kn_pred, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test, kn_pred)*100}%")
   
   ## Support Vector
   sv = SVC()
   sv.fit(x_train, y_train)
   sv_pred = sv.predict(x_test)
   print(f"f1 score = {f1_score(y_test, sv_pred, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test, sv_pred)*100}%")
   ```

6. **Balancing the Dataset using SMOTE**
   ```python
   smote = SMOTE()
   x_res, y_res = smote.fit_resample(x,y)
   y_res.value_counts().plot(kind='bar')
   
   x_train_, x_test_, y_train_, y_test_ = train_test_split(x_res, y_res, test_size=0.2)
   
   ## Random Forest
   rand_b = RandomForestClassifier()
   rand_b.fit(x_train_, y_train_)
   rand_pred_ = rand_b.predict(x_test_)
   print(f"f1 score = {f1_score(y_test_, rand_pred_, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test_, rand_pred_)*100}%")
   
   ## Logistic Regression
   log_b = LogisticRegression()
   log_b.fit(x_train_, y_train_)
   log_pred_ = log_b.predict(x_test_)
   print(f"f1 score = {f1_score(y_test_, log_pred_, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test_, log_pred_)*100}%")
   
   ## KNeighbors Classifier
   kn_b = KNeighborsClassifier()
   kn_b.fit(x_train_, y_train_)
   kn_pred_ = kn_b.predict(x_test_)
   print(f"f1 score = {f1_score(y_test_, kn_pred_, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test_, kn_pred_)*100}%")
   
   ## Support Vector
   sv_b = SVC()
   sv_b.fit(x_train_, y_train_)
   sv_pred_ = sv_b.predict(x_test_)
   print(f"f1 score = {f1_score(y_test_, sv_pred_, average='weighted')*100}%")
   print(f"Accuracy score = {accuracy_score(y_test_, sv_pred_)*100}%")
   ```

7. **Hyperparameter Tuning for RandomForestClassifier**
   ```python
   param_grid = {
       'bootstrap': [True, False],
       'max_depth': [15, 25, 30, 35],
       'n_estimators': [100, 250, 500]}
   
   rf = RandomForestClassifier()
   
   grid_search = GridSearchCV(estimator = rf, param_grid = param_grid)
   grid_search.fit(x_train_, y_train_)
   
   grid_search.best_params_
   
   pred = grid_search.predict(x_test_)
   
   print(f"f1 score = {f1_score(y_test_, pred)}")
   print(f"accuracy = {accuracy_score(y_test_, pred)}")
   ```

### Running the Code
1. Ensure all required libraries are installed.
2. Place the `train.csv` file in the same directory as the script.
3. Run the script using a Python interpreter.
