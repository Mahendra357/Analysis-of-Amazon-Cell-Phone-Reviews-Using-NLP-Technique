# **Model Development Phase**

In this phase, we preprocess the text data, convert it into numerical representations using NLP techniques, and split the data into training and testing sets. Scaling techniques are applied to normalize the data, ensuring optimal performance of machine learning models.

---

## **1. Text Preprocessing and Corpus Creation**

### **Objective:**
Prepare the textual data (`title` column) by removing noise and converting it into a clean, standardized format suitable for machine learning algorithms.  

### **Steps:**
- **Text Cleaning:** Removed all non-alphabetic characters using regex.
- **Lowercasing:** Converted all text to lowercase to avoid case sensitivity during analysis.
- **Tokenization and Stopword Removal:** Split each review into words and removed common stopwords (e.g., "and," "the").
- **Stemming:** Reduced words to their root form using the Porter Stemmer (e.g., "running" → "run").
- **Corpus Creation:** Combined the cleaned words back into complete reviews, stored in the `corpus` list.

### **Code:**
```python
corpus = []
stemmer = PorterStemmer()

for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data.iloc[i]['title'])  # Remove non-alphabetic characters
    review = review.lower().split()  # Convert to lowercase and split into words
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]  # Stem words and remove stopwords
    review = ' '.join(review)  # Rejoin words into a sentence
    corpus.append(review)
```

### **Outcome:**
- `corpus` now contains cleaned and stemmed reviews, ready for feature extraction.

---

## **2. Feature Extraction Using CountVectorizer**

### **Objective:**
Convert the cleaned text data in the `corpus` into numerical format using the **Bag-of-Words** model (CountVectorizer). This allows machine learning algorithms to process text as numeric data.  

### **Steps:**
- Initialized `CountVectorizer` with a maximum feature limit of 2500 to focus on the most frequent terms, reducing dimensionality.
- Transformed the text data into a sparse matrix `X`, where rows represent reviews, and columns represent word counts.

### **Code:**
```python
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()  # Convert text data to numerical format
y = data['verified'].values  # Target variable (sentiment)
```

### **Outcome:**
- `X` contains the numerical representation of reviews with dimensions `(rows, 2500 features)`.
- `y` contains binary sentiment labels (0 for negative, 1 for positive).

---

## **3. Saving the CountVectorizer Model**

### **Objective:**
Preserve the CountVectorizer model for future use to ensure consistency between training and testing phases.

### **Steps:**
- Created a directory called `Models` to store serialized models.
- Saved the CountVectorizer model as a `.pkl` file using the `pickle` library.

### **Code:**
```python
import os
import pickle

# Ensure the 'Models' directory exists
os.makedirs('Models', exist_ok=True)

# Save the CountVectorizer model
pickle.dump(cv, open('Models/countVectorizer.pkl', 'wb'))
```

### **Outcome:**
- The `countVectorizer.pkl` file was saved and can be loaded later for model inference.

---

## **4. Splitting the Dataset**

### **Objective:**
Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.

### **Steps:**
- Used `train_test_split` to split `X` and `y` into training (70%) and testing (30%) sets.
- Ensured randomization using a `random_state` for reproducibility.

### **Code:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=15)

# Display the shapes of training and testing sets
print(f"X train: {X_train.shape}")
print(f"y train: {y_train.shape}")
print(f"X test: {X_test.shape}")
print(f"y test: {y_test.shape}")
```

### **Outcome:**
- Training set (`X_train`, `y_train`) contains 70% of the data.
- Testing set (`X_test`, `y_test`) contains 30% of the data.
- The data is now ready for scaling and model training.

---

## **5. Data Normalization Using MinMaxScaler**

### **Objective:**
Scale the feature values in `X_train` and `X_test` to a range of 0 to 1 to prevent large values from dominating the machine learning model.

### **Steps:**
- Applied Min-Max Scaling on `X_train` and used the same scaler to transform `X_test`.
- Saved the scaler model for future use.

### **Code:**
```python
scaler = MinMaxScaler()

# Scale the training and testing sets
X_train_scl = scaler.fit_transform(X_train)
X_test_scl = scaler.transform(X_test)

# Save the scaler model
pickle.dump(scaler, open('Models/scaler.pkl', 'wb'))
```

### **Outcome:**
- Both training and testing sets were scaled to ensure consistency and improve model performance.
- The scaler model (`scaler.pkl`) was saved for reuse during inference.

---

## **6. Summary of Actions in Model Development Phase**

### **Accomplished Tasks:**
1. Preprocessed the review text by cleaning, stemming, and removing stopwords.
2. Converted the text data into a numerical format using the Bag-of-Words model (CountVectorizer).
3. Divided the dataset into training and testing sets for model evaluation.
4. Normalized the feature values using Min-Max Scaling.
5. Saved the CountVectorizer and scaler models for consistency in future use.

### **Next Steps:**
- Train machine learning models (e.g., Random Forest, Decision Tree) on the preprocessed data.
- Perform hyperparameter tuning and evaluate model performance using metrics like accuracy, precision, and recall.

---
Now let's proceed to the Model Training Phase....
