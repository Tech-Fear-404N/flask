import spacy
from fuzzywuzzy import fuzz
import pandas as pd
from flask import Flask, render_template, request
import speech_recognition as sr
import pyttsx3

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")

dataset = {}

def load_custom_dataset():
    try:
        df = pd.read_csv('./qna.csv')
        for index, row in df.iterrows():
            question = row["ques"].strip()
            answer = row["answer"].strip()
            dataset[question] = answer
    except Exception as e:
        print(f"Error loading custom dataset: {e}")

load_custom_dataset()

def chatbot_response(user_input):
    user_input = user_input.lower()

    user_input_tokens = nlp(user_input)
    best_match_question = None
    best_match_score = 0

    for question in dataset.keys():
        similarity_score = fuzz.ratio(user_input, question.lower())
        if similarity_score > best_match_score:
            best_match_score = similarity_score
            best_match_question = question

    if best_match_score > 51:
        return dataset[best_match_question]
    else:
        for ent in user_input_tokens.ents:
            if ent.label_ == "PERSON":
                return f"My name is {ent.text}."
        return "I'm not sure how to answer that."

def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # You can adjust the speaking rate as needed
    engine.say(text)
    engine.runAndWait()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.form.get("user_input")
    response = chatbot_response(user_input)
    speak_text(response)  # Speak the response
    return render_template("index.html", user_input=user_input, response=response)

@app.route("/voice_input", methods=["POST"])
def voice_input():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening for voice input...")
            audio = recognizer.listen(source)
            user_input = recognizer.recognize_google(audio)
            response = chatbot_response(user_input)
            speak_text(response)  # Speak the response
            return render_template("index.html", user_input=user_input, response=response)
    except sr.UnknownValueError:
        return render_template("index.html", user_input="Voice input not recognized.", response="I'm not sure how to answer that.")
    except sr.RequestError as e:
        return render_template("index.html", user_input="Could not request results from Google Speech Recognition.", response="I'm not sure how to answer that.")

if __name__ == "__main__":
    app.run(debug=True)
