from flask import Flask, render_template

app = Flask(__name__, template_folder='templates/html')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def chat_output():
    return render_template('index.html')

@app.route('/treinamento/', methods=["GET"])
def treinamento():
    return render_template('treinamento.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080)