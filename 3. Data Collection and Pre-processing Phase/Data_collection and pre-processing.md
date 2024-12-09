# **Data Collection and Preprocessing Phase**

In this phase, we focused on gathering the dataset, cleaning it to remove inconsistencies, and preparing it for analysis. This step is crucial to ensure the data is of high quality and structured correctly for modeling. Below is a detailed breakdown of the actions taken during this phase:

---

## **1. Data Collection**

The dataset containing Amazon cell phone reviews was imported into the project using the pandas library. This dataset includes attributes such as the review title, rating, helpful votes, and other features that can be used for sentiment analysis.

### **Steps Taken:**
- Loaded the dataset from the specified file path.
- Inspected the dimensions of the dataset, i.e., the number of rows and columns, to understand its size.
- Reviewed the first few records to explore the data structure and identify the relevant columns.

### **Code:**
```python
# Load the dataset from the specified path
data = pd.read_csv(r"C:/Users/Admin/Downloads/archive/20191226-reviews.csv")

# Print the shape of the dataset
print(f"Dataset shape : {data.shape}")

# Display the first few records to understand the structure
data.head()

# Print column names
print(f"Feature names : {data.columns.values}")
```

### **Findings:**
- The dataset consists of multiple features, with key columns like `title` (review text), `rating`, and `helpfulVotes` being relevant for sentiment analysis.
- The dataset size (number of rows and columns) was noted, ensuring it was sufficient for training a robust model.

---

## **2. Handling Missing Values**

To ensure the dataset's integrity, we identified and removed any records with missing values. Missing data can lead to errors or skewed results during model training, so addressing this early was critical.

### **Steps Taken:**
- Checked for null or missing values in each column.
- Displayed records with missing values to understand their impact.
- Removed rows with missing values to maintain data quality and consistency.

### **Code:**
```python
# Check for missing values in each column
data.isnull().sum()

# Display records with missing values
data[data['helpfulVotes'].isna() == True]

# Drop rows with missing values
data.dropna(inplace=True)

# Confirm the dataset's shape after removing missing values
print(f"Dataset shape after dropping null values : {data.shape}")
```

### **Findings:**
- Missing values were found in specific columns, such as `helpfulVotes`.
- After removing these rows, the dataset's shape slightly reduced, but the remaining data was clean and ready for analysis.

---

## **3. Feature Engineering - Adding a Review Length Column**

To better understand the textual data, a new column named `length` was created. This column measures the length of each review in the `title` column and provides insights into the verbosity of the feedback.

### **Steps Taken:**
- Calculated the character count for each review in the `title` column.
- Added this information as a new column, `length`.
- Verified the new column by comparing its values with the original review text.

### **Code:**
```python
# Create a new column with the length of the review text
data['length'] = data['title'].apply(len)

# Inspect the 10th record to confirm the new column
print(f"'title' column value: {data.iloc[10]['title']}")  # Original review
print(f"Length of review : {len(data.iloc[10]['title'])}")  # Calculated length
print(f"'length' column value : {data.iloc[10]['length']}")  # Length stored in the new column
```

### **Insights:**
- The `length` column provides valuable information, as longer reviews might indicate more thoughtful feedback, while shorter ones might suggest brevity or limited engagement.
- This feature can be used in further exploratory analysis or even as an input feature for modeling.

---

## **4. Analyzing Ratings Distribution**

Understanding the distribution of ratings is essential to assess whether the dataset is balanced or skewed toward certain ratings. This helps identify potential biases in the data.

### **Steps Taken:**
- Calculated the percentage distribution of each rating.
- Reviewed the results to determine whether the dataset favors higher or lower ratings.

### **Code:**
```python
# Calculate and display the percentage distribution of ratings
print(f"Rating value count - percentage distribution: \n{round(data['rating'].value_counts() / data.shape[0] * 100, 2)}")
```

### **Findings:**
- The dataset showed a clear distribution of ratings, revealing whether it was balanced or imbalanced.
- Such insights are important as they directly impact model training and performance. For example, if the data is heavily skewed toward positive ratings, the model might struggle to identify negative sentiments.

---

## **5. Defining Sentiments**

To simplify the sentiment analysis process, a binary column `verified` was created. This column classifies reviews into two categories:
- **Positive Reviews**: Ratings of 4 or higher.
- **Negative Reviews**: Ratings below 4.

### **Steps Taken:**
- Created a binary classification of reviews based on their ratings.
- Converted the binary values to integers (1 for positive, 0 for negative).
- Calculated the distribution of positive and negative reviews to verify data balance.

### **Code:**
```python
# Define positive and negative reviews based on the rating
data['verified'] = data['rating'].apply(lambda x: True if x >= 4 else False)

# Convert the Boolean values into integers (1 for positive, 0 for negative)
data['verified'] = data['verified'].astype(int)

# Display distinct values in the 'verified' column and their counts
print(f"Feedback value count: \n{data['verified'].value_counts()}")

# Calculate and display the percentage distribution of feedback
print(f"Feedback value count - percentage distribution: \n{round(data['verified'].value_counts() / data.shape[0] * 100, 2)}")
```

### **Insights:**
- The new column `verified` simplifies the target variable, making it easier to train classification models.
- The distribution of positive and negative feedback was reviewed to ensure there were enough records in both categories for balanced model training.

---

## **6. Summary of Preprocessing**

### **Actions Taken:**
1. Loaded the dataset and reviewed its structure.
2. Removed missing values to ensure data integrity.
3. Engineered a new `length` column to analyze the verbosity of reviews.
4. Analyzed the distribution of ratings to identify potential biases.
5. Created a binary sentiment column (`verified`) for simplified classification.

### **Insights Gained:**
- The dataset is now clean, consistent, and ready for text preprocessing.
- Features like review length and sentiment classification provide valuable inputs for exploratory analysis and modeling.
- The distribution of ratings and sentiments highlights potential areas to address, such as imbalances in the data.

---

## **Next Steps**

In the next phase, we will:
1. Perform text preprocessing, including tokenization, stemming, and stopword removal.
2. Extract features using NLP techniques such as CountVectorizer or TF-IDF.
3. Begin exploratory data analysis to visualize and understand patterns in the dataset.


Now the data is cleaned and can be used to create models ...
