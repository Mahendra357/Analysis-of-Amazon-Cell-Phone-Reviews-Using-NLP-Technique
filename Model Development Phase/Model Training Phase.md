# **Model Training Phase**

In this phase, we train multiple machine learning models on the preprocessed data, evaluate their performance using accuracy scores, and fine-tune hyperparameters using techniques like Grid Search. This helps us determine the best-performing model for predicting sentiment in reviews.

---

## **1. Training Random Forest Classifier**

### **Objective:**
Train the Random Forest model on the scaled training data (`X_train_scl`, `y_train`) and evaluate its performance.

### **Steps:**
- We start by training a **Random Forest Classifier** on the training data using the `fit()` method.
- After training, we assess the model's accuracy on both the training set and the test set.

### **Code:**
```python
# Fitting the Random Forest Classifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train_scl, y_train)

# Calculate accuracy on training and testing data
print("Training Accuracy :", model_rf.score(X_train_scl, y_train))
print("Testing Accuracy :", model_rf.score(X_test_scl, y_test))
```

### **Outcome:**
- The training accuracy provides insight into how well the model has learned from the training data.
- The testing accuracy indicates how well the model generalizes to new, unseen data.

---

## **2. Predicting and Evaluating the Model with Confusion Matrix**

### **Objective:**
Evaluate the predictions made by the trained model on the test data and visualize the results using a confusion matrix.

### **Steps:**
- We use the trained model to make predictions (`y_preds`) on the test set (`X_test_scl`).
- A **confusion matrix** is generated to show the performance of the classification model by displaying true positives, true negatives, false positives, and false negatives.
- The confusion matrix is visualized using `ConfusionMatrixDisplay`.

### **Code:**
```python
y_preds = model_rf.predict(X_test_scl)

# Generating confusion matrix
cm = confusion_matrix(y_test, y_preds)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_rf.classes_)
cm_display.plot()
plt.show()
```

### **Outcome:**
- The confusion matrix helps identify whether the model is making mistakes, particularly false positives (incorrectly classifying a negative review as positive) and false negatives (incorrectly classifying a positive review as negative).

---

## **3. Model Evaluation Using Cross-Validation**

### **Objective:**
Assess the model's stability and robustness by evaluating it using **cross-validation**.

### **Steps:**
- We perform **cross-validation** with 10 folds using `cross_val_score`, which splits the data into 10 parts and trains/evaluates the model 10 times, each time using a different subset of the data.
- This gives a more reliable estimate of model performance.

### **Code:**
```python
accuracies = cross_val_score(estimator=model_rf, X=X_train_scl, y=y_train, cv=10)
print("Accuracy :", accuracies.mean())  # Mean accuracy
print("Standard Variance :", accuracies.std())  # Variance of accuracy across folds
```

### **Outcome:**
- The **mean accuracy** provides an overall estimate of how well the model is expected to perform.
- **Standard variance** tells us the consistency of the model’s performance across different data splits.

---

## **4. Hyperparameter Tuning Using Grid Search**

### **Objective:**
Optimize the model's performance by tuning the hyperparameters (e.g., `max_depth`, `n_estimators`) using **Grid Search**.

### **Steps:**
- We define a set of hyperparameters (`params`) to explore and tune.
- We use **GridSearchCV**, which systematically tests different combinations of hyperparameters to find the best-performing configuration.
- The model is evaluated on the training data using cross-validation, and the best parameters are displayed.

### **Code:**
```python
params = {
    'bootstrap': [True],
    'max_depth': [80, 100],
    'min_samples_split': [8, 12],
    'n_estimators': [100, 300]
}

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=model_rf, param_grid=params, cv=2, verbose=0, return_train_score=True)
grid_search.fit(X_train_scl, y_train.ravel())

# Display best parameters and cross-validation results
print("Best Parameter Combination : {}".format(grid_search.best_params_))
print("Cross validation mean accuracy on train set : {}".format(grid_search.cv_results_['mean_train_score'].mean()*100))
print("Cross validation mean accuracy on test set : {}".format(grid_search.cv_results_['mean_test_score'].mean()*100))
```

### **Outcome:**
- We obtain the **best hyperparameters** for the Random Forest model, ensuring the model is as accurate as possible.
- **Cross-validation mean accuracy** on both train and test sets provides an overview of how well the model performs across all data subsets.

---

## **5. Training Decision Tree Classifier**

### **Objective:**
Train a **Decision Tree Classifier** to compare its performance with the Random Forest model.

### **Steps:**
- The model is trained using the `fit()` method.
- Accuracy is calculated on both training and testing data.
- We visualize the predictions using a confusion matrix.

### **Code:**
```python
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train_scl, y_train)

# Accuracy on training and testing data
print("Training Accuracy :", model_dt.score(X_train_scl, y_train))
print("Testing Accuracy :", model_dt.score(X_test_scl, y_test))

# Predictions and confusion matrix
y_preds = model_dt.predict(X_test)
cm = confusion_matrix(y_test, y_preds)
print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_dt.classes_)
cm_display.plot()
plt.show()
```

### **Outcome:**
- We compare the performance of the **Decision Tree** with the Random Forest model to check if it performs better, worse, or similarly.

---

## **6. Training XGBoost Classifier**

### **Objective:**
Train the **XGBoost** classifier to evaluate its performance for sentiment classification.

### **Steps:**
- The model is trained on the scaled training data.
- We calculate accuracy on both training and test data and visualize the confusion matrix.

### **Code:**
```python
from xgboost import XGBClassifier

model_xgb = XGBClassifier()
model_xgb.fit(X_train_scl, y_train)

# Accuracy on training and testing data
print("Training Accuracy :", model_xgb.score(X_train_scl, y_train))
print("Testing Accuracy :", model_xgb.score(X_test_scl, y_test))

# Predictions and confusion matrix
y_preds = model_xgb.predict(X_test)
cm = confusion_matrix(y_test, y_preds)
print(cm)
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_xgb.classes_)
cm_display.plot()
plt.show()

# Saving the XGBoost classifier model
pickle.dump(model_xgb, open('Models/model_xgb.pkl', 'wb'))
```

### **Outcome:**
- The **XGBoost** model is trained, evaluated, and its performance is compared against the Random Forest and Decision Tree models.
- The **XGBoost model** is saved for future inference.

---

## **7. Summary of Actions in Model Training and Tuning Phase**

### **Accomplished Tasks:**
1. Trained multiple models (Random Forest, Decision Tree, XGBoost).
2. Evaluated models using accuracy scores, confusion matrices, and cross-validation.
3. Tuned hyperparameters for the Random Forest model using Grid Search.
4. Compared the performance of different classifiers to determine the best model.
5. Saved the final XGBoost model for future use.

### **Next Steps:**
- After finalizing the best model, we would integrate it into an application or system that can predict sentiments on new reviews.
- Perform model deployment and inference tasks for real-world application.

---
Now Let's check how the prediction model will be made using html and flask
