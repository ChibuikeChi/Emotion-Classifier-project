# Emotion-Classifier-project
The Emotion Classifier is an intelligent system designed to automatically analyze text and determine the underlying human emotional state expressed within that text.
Instead of a simple positive or negative label (like in sentiment analysis), this system differentiates between a range of specific human emotions.

Input: Any text string (e.g., "I am thrilled to be working on this problem," or "I am quite worried about the deadline.").
Output: A predicted emotional label, such as: Joy, Fear, Sadness, Anger, Love, Surprise

In short, this project teaches a computer to 'read between the lines' and understand the emotional context of human language.

How It Was Built (The Workflow)
- Data Acquisition: A large dataset of text (e.g., social media comments, headlines) was collected, where each piece of text was manually labeled with a specific emotion.
- Text Preprocessing: The text was cleaned (e.g., removing punctuation, converting to lowercase) and tokenized (broken down into individual words).
- Feature Extraction: The raw words were converted into a format a machine learning model can understand, typically using a technique like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings.
- Model Training: A supervised learning algorithm was trained on the processed data to learn the association between word patterns and specific emotion labels.
- Prediction: The trained model is then used to predict the emotion for any new, unseen piece of text.

Key Libraries Used
Building a machine learning project like this requires a stack of specialized Python libraries:

  Library  (Category)                  Core Function in the Project
- Scikit-learn (Machine Learning): Used for the classifier model itself (e.g.,LogisticRegression, RandomForestClassifier), splitting the data, and evaluating performance.
- NLTK (Natural Language Toolkit): NLP  FundamentalsUsed for low-level text preprocessing, such as tokenization, stemming, and removing stop words (like "the," "a," "is").
- Pandas (Data Management): Used for loading, manipulating, and cleaning the large dataset of text and emotion labels.
- NumPy (Numerical Operations): Provides fast, efficient numerical arrays necessary for handling the high-dimensional data generated during feature extraction.
- TfidfVectorizer (from sklearn): Feature ExtractionCrucial tool used to convert raw text into numerical features (TF-IDF vectors) that the machine learning model can process.

The application is deployed using Streamlit, allowing users to interact with the trained model through a web interface.

Step 1: Project Setup

Ensure the following files are present in the project directory:

  EmotionDotio.py
  emotion_lstm_model.keras
  emotion_tokenizer.json
  emotion_features.joblib
  requirements.txt

The EmotionDotio.py file contains the Streamlit application code.

Step 2: Create and Activate a Virtual Environment

Open Command Prompt or Terminal and navigate to the project folder:  cd path\to\emotion_project
Create a virtual environment:  python -m venv stenv
Activate the environment:  stenv\Scripts\activate

Once activated, (stenv) will appear in the terminal.

Step 3: Install Required Dependencies

Install all required libraries:
pip install -r requirements.txt

Typical dependencies include:
streamlit
tensorflow
numpy
joblib

Step 4: Run the Streamlit Application Locally

Start the application by running:
streamlit run EmotionDotio.py
Streamlit will automatically open the application in a browser at:
http://localhost:8501
Users can now enter text and receive emotion predictions.

Step 5: Deploy the App Using Streamlit Community Cloud

Push the project to a GitHub repository
Visit https://share.streamlit.io
Sign in with GitHub
Select the repository and the main file (EmotionDotio.py)
Click Deploy
Streamlit will build the application and provide a public URL.

Step 6: Access the Deployed App

Once deployed, the app can be accessed from any browser using the provided Streamlit URL.
No local Python installation is required for end users.
