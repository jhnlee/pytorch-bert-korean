from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'hello world!'

@app.route('/info')
def info():
    return 'Info'
