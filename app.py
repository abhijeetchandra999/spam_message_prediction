from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    user_input = None
    if request.method == "POST":
        user_input = request.form.get("comment")
        if user_input:
            transformed_text = vectorizer.transform([user_input])
            prediction = model.predict(transformed_text)[0]
            result = "✅ Spam" if prediction == 1 else "✅ Not Spam"
    return render_template("index.html", result=result, user_input=user_input)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
