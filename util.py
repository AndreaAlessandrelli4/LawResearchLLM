import os
import re
import json
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import weaviate
import torch
from weaviate.gql.get import HybridFusion
#from weaviate.gql.query import HybridFusion
import pdfplumber
import fitz  # PyMuPDF
import docx

#def extract_text_from_pdf_with_fitz(uploaded_file):
#    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
#    text = ""
#    for page in doc:
#        text += page.get_text("text")
#    return text

def extract_text_from_docx(uploaded_file):
    try:
        doc = docx.Document(uploaded_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        return f"Errore durante l'elaborazione del DOCX: {e}"
        

def extract_text_from_pdf(uploaded_file):
    """Estrae il testo da un PDF usando pdfplumber."""
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_txt(uploaded_file):
    """Legge il contenuto di un file di testo."""
    return uploaded_file.read().decode("utf-8")


def cleaning(text):
    #text=re.sub(r'\s+',' ',text)
    text=re.sub(r"""Pagina \d+ di \d+|pagina \d+ di \d+|Pag. \d+ di \d+|
                                   Pag. \d+|Pagina \d+|pagina \d+""",'',text)
    return text



# Funzione per generare gli embedding
def generate_embeddings(entry, model):
    embedding = model.encode(entry,  convert_to_numpy=True, normalize_embeddings=True)
    return np.array([i for i in embedding])


def collect_paths(data):
    paths = []

    if isinstance(data, dict):
        for key, value in data.items():
            if key == "path":
                paths.append(value[0])
            else:
                paths.extend(collect_paths(value))

    elif isinstance(data, list):
        for item in data:
            paths.extend(collect_paths(item))

    return paths


def scores(text):
    # Trova tutti i valori dopo 'normalized score:'
    matches = re.findall(r'normalized score:\s*([\d.]+)', text)
    return [float(x) for x in matches]




# Funzione per eseguire la query su Weaviate
def query_weaviate(client, query, num_max, alpha, filters, search_prop=["testo_parziale", 'summary'],
                   retrieved_proop=["riferimenti_legge","testo_parziale", 'estrazione_mistral', 'summary', 'testo_completo', 'id_originale']):
    more_prop = collect_paths(filters)
    if len(more_prop)>=1:
        retrieved_proop = [*retrieved_proop, *more_prop]
        response = (
            client.query
            .get("TestoCompleto", retrieved_proop)
            .with_hybrid(
                query=query,
                vector=list(generate_embeddings(query)),
                properties=search_prop,
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
            ).with_where(filters).with_additional(["score", "explainScore"]).do())
    else:
        response = (
            client.query
            .get("TestoCompleto", retrieved_proop)
            .with_hybrid(
                query=query,
                vector=list(generate_embeddings(query)),
                properties=search_prop,
                alpha=alpha,
                fusion_type=HybridFusion.RELATIVE_SCORE,
            ).with_additional(["score", "explainScore"]).do())
        
    ids = []
    risposta_finale = []
    try:
        for i in response['data']['Get']["TestoCompleto"]:
            ids_temp = i['id_originale']
            if ids_temp not in ids:
                ids.append(ids_temp)
                diz = {}
                try:
                    diz['query_score'] = float(i["_additional"]["score"])
                except:
                    diz['query_score'] = i["_additional"]["explainScore"]
                diz['id_originale'] = i['id_originale']
                diz['summary'] = i['summary']
                diz['testo_completo'] = i['testo_completo']
                diz['metaDati'] = json.loads(i['estrazione_mistral'])
                diz['riferimenti_legge']= i["riferimenti_legge"]
                risposta_finale.append(diz)
                risposta_finale=risposta_finale[0:num_max]
    except:
        risposta_finale=response
    return risposta_finale



