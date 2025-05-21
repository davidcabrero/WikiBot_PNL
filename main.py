import nltk
import numpy as np
import random
import string
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurar Wikipedia en español
wikipedia.set_lang("es")

# Descargar recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')

# 1. Tokenización

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

# 2. Lematización

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

# 3. Saludos

SALUDOS_IN = ["hola", "buenas", "qué tal", "buenos días", "hey", "saludos"]
SALUDOS_OUT = ["Hola", "Hola, ¿cómo estás?", "¡Saludos!", "Encantado de ayudarte", "¿En qué te puedo ayudar?"]

def generar_saludo(frase):
    for palabra in frase.split():
        if palabra.lower() in SALUDOS_IN:
            return random.choice(SALUDOS_OUT)

# 4. Búsqueda en Wikipedia

def buscar_wikipedia(pregunta):
    try:
        resultado = wikipedia.summary(pregunta, sentences=2)
        return resultado
    except wikipedia.exceptions.DisambiguationError as e:
        return f"La pregunta es ambigua. ¿Quizás te refieres a: {', '.join(e.options[:5])}?"
    except wikipedia.exceptions.PageError:
        return "No encontré nada relacionado en Wikipedia."
    except Exception as e:
        return "Error al buscar en Wikipedia."

# 5. Generar Respuesta

def responder(respuesta_usuario):
    # Intentar con Wikipedia primero
    resultado_wiki = buscar_wikipedia(respuesta_usuario)
    if "no encontré nada" in resultado_wiki.lower() or "ambigua" in resultado_wiki.lower() or "error" in resultado_wiki.lower():
        # Respuesta basada en TF-IDF si Wikipedia no ayuda
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

# 6. Chatbot principal desde la app
def chatbot_general(mensaje):
    if mensaje.lower() == 'gracias':
        return "¡De nada!"
    elif generar_saludo(mensaje) is not None:
        return generar_saludo(mensaje)
    else:
        return responder(mensaje)

# Para pruebas por consola
def chatbot_console():
    print("WikiBot: Me llamo WikiBot. ¡Pregúntame lo que quieras!")
    print("Escribe 'salir' para terminar la conversación.")
    while True:
        info_usuario = input("TÚ> ").lower()
        if info_usuario == 'salir':
            print("WikiBot> ¡Hasta luego!")
            break
        elif info_usuario == 'gracias':
            print("WikiBot> ¡De nada!")
        elif generar_saludo(info_usuario) is not None:
            print("WikiBot> " + generar_saludo(info_usuario))
        else:
            print("WikiBot> " + responder(info_usuario))

# Para probar directamente este script
if __name__ == "__main__":
    chatbot_console()
