from flask import Flask, render_template, request, jsonify
import main

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    mensaje = request.json.get("mensaje")
    idioma = request.json.get("idioma")
    resultado = main.chatbot_general(mensaje, idioma)
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True)
