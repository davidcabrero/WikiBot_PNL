import nltk
import numpy as np
import random
import string
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from textblob import TextBlob
from transformers import MarianMTModel, MarianTokenizer

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# === Carga y Preprocesado ===

def preprocesar(fichero):
    with open(fichero, 'r', errors='ignore') as f:
        data_raw = f.read().lower()
    return data_raw

datos_prep = preprocesar('info.txt')

def tokenizar_frases(datos):
    return nltk.sent_tokenize(datos)

def tokenizar_palabras(datos):
    return nltk.word_tokenize(datos)

frases_tokens = tokenizar_frases(datos_prep)
palabras_tokens = tokenizar_palabras(datos_prep)

# === Lematización ===

def lanzarLemitizador():
    return nltk.stem.WordNetLemmatizer()

mi_lemitizador = lanzarLemitizador()

def LemitizarTokens(ltokens, lemit):
    return [lemit.lemmatize(token) for token in ltokens]

def obtener_signos_puntuacion():
    return dict((ord(punct), None) for punct in string.punctuation)

signos_punt_a_quitar = obtener_signos_puntuacion()

def LemNormalizada(texto):
    return LemitizarTokens(nltk.word_tokenize(texto.lower().translate(signos_punt_a_quitar)), mi_lemitizador)

# === Saludos ===

SALUDOS_IN_es = ["hola", "buenas", "qué tal", "buenos días", "hey", "saludos"]
SALUDOS_OUT_es = ["Hola", "Hola, ¿cómo estás?", "¡Saludos!", "Encantado de ayudarte", "¿En qué te puedo ayudar?"]

SALUDOS_IN_en = ["hello", "hey", "how are you?", "good morning", "hi", "What's up?"]
SALUDOS_OUT_en = ["Hello", "Hello, How are you?", "¡Hi, there!", "Nice to help you", "How can I help you?"]

SALUDOS_IN_ast = ["hola", "bones", "qué tal", "bonos díes", "hey", "bienveníu"]
SALUDOS_OUT_ast = ["Hola", "Hola, ¿cómo tas?", "¡Saludos!", "Encantau d'ayudate'", "¿En qué pueu ayudate güei?"]

def generar_saludo(frase, idioma):

    if idioma == "es":
        SALUDOS_IN = SALUDOS_IN_es
        SALUDOS_OUT = SALUDOS_OUT_es
    elif idioma == "en":
        SALUDOS_IN = SALUDOS_IN_en
        SALUDOS_OUT = SALUDOS_OUT_en
    elif idioma == "ast":
        SALUDOS_IN = SALUDOS_IN_ast
        SALUDOS_OUT = SALUDOS_OUT_ast
        
    for palabra in frase.split():
        if palabra.lower() in SALUDOS_IN:
            return random.choice(SALUDOS_OUT)

# === Wikipedia ===

def buscar_wikipedia(pregunta, idioma="es"):
    # Configuramos el idioma de Wikipedia según la elección del usuario
    wikipedia.set_lang(idioma)
    try:
        resultado = wikipedia.summary(pregunta, sentences=2)
        return resultado
    except wikipedia.exceptions.DisambiguationError as e:
        return f"La pregunta es ambigua. ¿Quizás te refieres a: {', '.join(e.options[:5])}?"
    except wikipedia.exceptions.PageError:
        return "No encontré nada relacionado en Wikipedia."
    except Exception as e:
        return "Error al buscar en Wikipedia."

# === TF-IDF ===

def responder(respuesta_usuario, idioma):
    resultado_wiki = buscar_wikipedia(respuesta_usuario, idioma)
    if "no encontré nada" in resultado_wiki.lower() or "ambigua" in resultado_wiki.lower() or "error" in resultado_wiki.lower():
        bot_respuesta = ''
        frases_tokens.append(respuesta_usuario)
        tfidfVec = TfidfVectorizer(tokenizer=LemNormalizada, stop_words='english')
        tfidfVec_ajustado = tfidfVec.fit_transform(frases_tokens)
        valores = cosine_similarity(tfidfVec_ajustado[-1], tfidfVec_ajustado)
        indice = valores.argsort()[0][-2]
        valores_aplanada = valores.flatten()
        valores_aplanada.sort()
        valor_similitud_buscado = valores_aplanada[-2]
        if valor_similitud_buscado == 0:
            bot_respuesta = "Lo siento, no entendí eso."
        else:
            bot_respuesta = frases_tokens[indice]
        frases_tokens.remove(respuesta_usuario)
        return bot_respuesta
    else:
        return resultado_wiki

# === Análisis de Sentimiento ===

def analizar_sentimiento(texto):
    blob = TextBlob(texto)
    polaridad = blob.sentiment.polarity
    if polaridad > 0:
        return "positivo"
    elif polaridad < 0:
        return "negativo"
    else:
        return "neutral"


# === Chatbot general ===

def chatbot_general(mensaje, idioma):
    if mensaje.lower() == 'gracias':
        respuesta = "¡De nada!"
    elif generar_saludo(mensaje, idioma) is not None:
        respuesta = generar_saludo(mensaje, idioma)
    else:
        respuesta = responder(mensaje, idioma)

    sentimiento = analizar_sentimiento(mensaje)

    return {
        "respuesta": respuesta,
        "sentimiento": sentimiento
    }
