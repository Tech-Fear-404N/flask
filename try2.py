import spacy
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from fuzzywuzzy import fuzz
import pandas as pd
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load the spaCy English language model
nlp = spacy.load("en_core_web_sm")

# Initialize an empty dictionary for the dataset
dataset = {}

# Function to load your custom dataset from a CSV file
def load_custom_dataset():
    try:
        # Read the CSV file with questions and answers
        df = pd.read_csv('./qna.csv') # Corrected the CSV filename
        
        # Assuming your CSV file has columns named "ques" and "answer"
        for index, row in df.iterrows():
            question = row["ques"].strip()  # Remove leading/trailing spaces
            answer = row["answer"].strip()  # Remove leading/trailing spaces
            dataset[question] = answer
    except Exception as e:
        print(f"Error loading custom dataset: {e}")

# Load your custom dataset from a CSV file (replace 'your_dataset.csv' with your CSV file)
load_custom_dataset()

# Define a function to get chatbot responses
def chatbot_response(user_input):
    user_input = user_input.lower()
    
    # Tokenize the user input using NLTK
    user_input_tokens = word_tokenize(user_input)
    
    # Initialize variables to track the best match
    best_match_question = None
    best_match_score = 0
    
    # Loop through the dataset questions and calculate fuzzy match scores
    for question in dataset.keys():
        similarity_score = fuzz.ratio(user_input, question.lower())
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_question = question
    
    # Check if the best match score is above a certain threshold
    if best_match_score > 51:  # Adjust the threshold as needed
        return dataset[best_match_question]
    else:
        # Use spaCy to extract named entities (e.g., names)
        user_input_doc = nlp(user_input)
        for ent in user_input_doc.ents:
            if ent.label_ == "PERSON":
                return f"My name is {ent.text}."
        
        return "I'm not sure how to answer that."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    response = chatbot_response(user_input)
    return render_template("index.html", user_input=user_input, response=response)

if __name__ == "__main__":
    app.run(debug=True)
