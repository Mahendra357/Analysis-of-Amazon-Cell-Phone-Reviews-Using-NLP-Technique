# Project Overview: Amazon Cell Phone Review Sentiment Analysis Using NLP

This project is a web application built with Flask that performs sentiment analysis on Amazon cell phone reviews, leveraging natural language processing (NLP) techniques. The application enables users to enter review text and receive a real-time classification of sentiment as either positive or negative, based on a trained machine learning model

## Project Structure

- **/Dataset/**: Contains the structured data of amazon cellphone reviews in csv format.
- **/templates/**: Contains HTML templates for user input forms and sentiment analysis result display.
- **/model/**: Directory for storing the trained sentiment analysis model and vectoriser and scaler.
- **api.py**: Main Flask application file that handles routes, prediction logic, and user interactions.
- **requirements.txt**: List of required libraries and dependencies for the project.

## Setup Instructions

Step 1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

Step 2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

Step 3. **Run the Application**  
   Start the Flask application with:
   ```bash
   python api.py
   ```

Step 4: **The app will run on port 5000.** 
   The app will run on a default port made by html.
   ```bash
   localhost:5000
   ```
## Usage

1. **Input**: Users submit an Amazon cell phone review via the application's input field.
2. **Prediction**:  Selecting the "Predict" button initiates the sentiment analysis on the submitted review.
3. **Result**:  The application returns and displays whether the review sentiment is classified as positive or negative.

## Key Features

- **Interactive Sentiment Analysis**: User-friendly interface designed to facilitate straightforward review sentiment analysis.
- **NLP-Driven Model**:Trained on a substantial dataset of Amazon cell phone reviews, the model is optimized to deliver reliable and accurate sentiment predictions.
- **Scalable  Architecture**: The model’s framework supports future expansion, allowing for easy retraining or fine-tuning with updated datasets as necessary.

## Requirements

The project requires Python 3.x and the dependencies listed in `requirements.txt`. Key libraries include Flask for the web framework, scikit-learn or xgboost for NLP modeling, and NLTK or seaborn for natural language processing.

## Future Enhancements

- **Expanded Sentiment Categories**: Introduce additional sentiment classifications, such as "neutral" or "mixed," to enhance analysis depth.
- **Model Optimization**: Improve prediction accuracy by performing hyperparameter tuning and incorporating expanded datasets.
- **Enhanced User Interface**: Enrich the UI by integrating visual representations, such as bar charts or pie charts, for a more comprehensive sentiment breakdown.
