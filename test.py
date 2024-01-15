# Example with Flask
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Render the main page
@app.route('/')
def home():
    return ("hii")

if __name__ == '__main__':
    app.run(debug=True)
