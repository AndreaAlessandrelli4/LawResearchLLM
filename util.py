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
from huggingface_hub import snapshot_download
from pathlib import Path
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

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
def query_weaviate(client, model, query, num_max, alpha, filters, search_prop=["testo_parziale", 'summary'],
                   retrieved_proop=["riferimenti_legge","testo_parziale", 'estrazione_mistral', 'summary', 'testo_completo', 'id_originale']):
    more_prop = collect_paths(filters)
    if len(more_prop)>=1:
        retrieved_proop = [*retrieved_proop, *more_prop]
        response = (
            client.query
            .get("TestoCompleto", retrieved_proop)
            .with_hybrid(
                query=query,
                vector=list(generate_embeddings(query, model)),
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
                vector=list(generate_embeddings(query, model)),
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



def prompt_query(query):
    prompt = """Sei un assistente intelligente che aiuta gli utenti a confrontare e ricercare sentenze all'interno di un database giuridico. 
Ogni sentenza è associata a una serie di metadati che descrivono vari aspetti del caso legale. 
Il tuo compito è rispondere a query testuali dell'utente restituendo un oggetto JSON con i metadati pertinenti. 
Ogni metadato ha valori ben definiti che devono essere utilizzati esclusivamente, senza modifiche o aggiustamenti.

**Istruzioni:**
1. Per ciascun campo, rispetta rigorosamente i valori predefiniti che ti vengono forniti nel JSON sottostante. 
Ogni campo ha un set di valori possibili (per le stringhe) o un intervallo di numeri (per i numeri), che **deve essere rispettato** senza alcuna variazione. 
Se un dato non è pertinente alla query dell'utente, usa il valore `"NON SPECIFICATO"` per le stringhe o 0 per i numeri.

2. **Valori stringa:** Per ciascun campo che accetta una stringa, scegli il valore che corrisponde a uno dei valori presenti nelle parentesi quadre nella struttura `filters_structure`. Non utilizzare valori al di fuori di quelli proposti.

3. **Valori numerici:** Per i campi che accettano numeri, utilizza valori intero per indicare l'intervallo in cui cadono i valori. Usa solo valori che rientra nell'intervallo specificato (minimo e massimo). Se non è specificato un numero nella query dell'utente, usa [0,0].

4. **Rispettare esclusivamente i valori predefiniti:** Ogni campo nel JSON ha un insieme limitato di valori possibili, che **deve essere rispettato** senza modifiche. Se una determinata chiave non viene menzionata nella query, inserisci `"NON SPECIFICATO"`.

**Struttura JSON (filters_structure):**
```json
{
    "info_generali_sentenza__tipo_separazione": ["giudiziale/contenzioso", "consensuale/congiunto", "NON SPECIFICATO"] (descrizione: "riporta il tipo di separazione tra i coniugi"),
    "info_generali_sentenza__modalita_esaurimento": ["omologazione", "accoglimento", "conciliazione", "cambiamento di rito", "archiviazione", "estinzione", "NON SPECIFICATO"] (descrizione: "riporta le modalita di esaurimento"),
    "info_generali_sentenza__violenza": ["Sì", "No", "NON SPECIFICATO"] (descrizione: "sono segnalati episodi di violenza?"),
    "info_generali_sentenza__abusi": ["Sì", "No", "NON SPECIFICATO"]  (descrizione: "sono segnalati episodi di abusi (escludi episodi di violenza che rientrano nel punto precedente)"),
    "dettagli_matrimonio__rito": ["religioso", "civile", "NON SPECIFICATO"] (descrizione: "rito con cui è stato celebrato il matrimonio"),
    "dettagli_matrimonio__regime_patrimoniale": ["comunione dei beni", "separazione dei beni", "NON SPECIFICATO"] (descrizione: "regime patrimoniale prima della separazione"),
    "dettagli_figli__numero_totale_di_figli": [0, 10] (descrizione:'numero totale di figli?'),
    "dettagli_figli__numero_di_figli_minorenni": [0, 10] (descrizione:'numero di figli minorenni'),
    "dettagli_figli__numero_di_figli_maggiorenni_economicamente_indipendenti": [0, 10](descrizione:'numero di figli maggiorenni economicamente autosufficienti'),
    "dettagli_figli__numero_di_figli_maggiorenni_non_economicamente_indipendenti": [0, 10](descrizione:'numero di figli maggiorenni NON economicamente autosufficienti'),
    "dettagli_figli__numero_di_figli_portatori_di_handicap": [0, 10] (descrizione:'numero di figli con handicap'),
    "dettagli_figli__tipo_di_affidamento": ["esclusivo al padre", "esclusivo alla madre", "esclusivo a terzi", "condiviso con prevalenza al padre", "condiviso con prevalenza alla madre", "condiviso con frequentazione paritetica", "NON SPECIFICATO"] (descrizione:'Specificare il tipo di affidamento dei figli')",
    "dettagli_figli__contributi_economici__contributi_per_il_mantenimento_figli": ["Sì", "No", "NON SPECIFICATO"] (descrizione:'Specificare se è presente un contributo che un coniuge deve versare per il mantenimento dei figli')",
    "dettagli_figli__contributi_economici__importo_assegno_per_il_mantenimento_figli": [0, 5000] (descrizione:'Specificare, qualora fosse presente un assegno di mantenimento per i figli, a quanto ammonta in EURO senza decimali')",
    "dettagli_figli__contributi_economici__obbligato_al_mantenimento": ["padre", "madre", "NON SPECIFICATO"] (descrizione:'Specificare, qualora fosse presente un assegno di mantenimento per i figli, chi è obbligato a versare tale contributo')",
    "dettagli_figli__contributi_economici__beneficiario_assegno_per_mantenimento_figli": ["direttamente ai figli", "all’altro genitore", "diviso", "NON SPECIFICATO"](descrizione:'Specificare, qualora fosse presente un assegno di mantenimento per i figli, chi è il beneficiario di tale contributo')",
    "dettagli_figli__contributi_economici__modalita_pagamento_assegno_di_mantenimento_del_coniuge": ["mensile", "una tantum", "NON SPECIFICATO"](descrizione:'Specificare, qualora fosse presente un contributo, le modalità di versamento/le tempistiche')",
}
```

**1° Esempio di query dell'utente:**
- "Cerca sentenze con una separazione consensuale, violenza domestica, e con almeno due figli affidati esclusivamente alla madre che percepisce un assegno di mantenimento di 250 euro."

**1° Esempio di risposta attesa (rispettando i valori predefiniti):**
```JSON
{
    "info_generali_sentenza__tipo_separazione": "consensuale/congiunto",
    "info_generali_sentenza__modalita_esaurimento": "NON SPECIFICATO",
    "info_generali_sentenza__violenza": "Sì",
    "info_generali_sentenza__abusi": "NON SPECIFICATO",
    "dettagli_matrimonio__rito": "NON SPECIFICATO",
    "dettagli_matrimonio__regime_patrimoniale": "NON SPECIFICATO",
    "dettagli_figli__numero_totale_di_figli": [2,10],
    "dettagli_figli__numero_di_figli_minorenni": [0,10],
    "dettagli_figli__numero_di_figli_maggiorenni_economicamente_indipendenti": [0,10],
    "dettagli_figli__numero_di_figli_maggiorenni_non_economicamente_indipendenti": [0,10]',
    "dettagli_figli__numero_di_figli_portatori_di_handicap": [0,10],
    "dettagli_figli__tipo_di_affidamento": "esclusivo alla madre",
    "dettagli_figli__contributi_economici__contributi_per_il_mantenimento_figli": "Sì",
    "dettagli_figli__contributi_economici__importo_assegno_per_il_mantenimento_figli": [0,250],
    "dettagli_figli__contributi_economici__obbligato_al_mantenimento": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__beneficiario_assegno_per_mantenimento_figli": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__modalita_pagamento_assegno_di_mantenimento_del_coniuge": "NON SPECIFICATO"
}
```


**2° Esempio di query dell'utente:**
- "Cerca sentenze che coinvolgono figli unici in cui siano presenente al abusi del marito sulla moglie  e abuso di alcol."

**2° Esempio di risposta attesa (rispettando i valori predefiniti):**
```JSON
{
    "info_generali_sentenza__tipo_separazione": "NON SPECIFICATO",
    "info_generali_sentenza__modalita_esaurimento": "NON SPECIFICATO",
    "info_generali_sentenza__violenza": "NON SPECIFICATO",
    "info_generali_sentenza__abusi": "Sì",
    "dettagli_matrimonio__rito": "NON SPECIFICATO",
    "dettagli_matrimonio__regime_patrimoniale": "NON SPECIFICATO",
    "dettagli_figli__numero_totale_di_figli": [0,1],
    "dettagli_figli__numero_di_figli_minorenni": [0,10],
    "dettagli_figli__numero_di_figli_maggiorenni_economicamente_indipendenti": [0,10],
    "dettagli_figli__numero_di_figli_maggiorenni_non_economicamente_indipendenti": [0,10]',
    "dettagli_figli__numero_di_figli_portatori_di_handicap": [0,10],
    "dettagli_figli__tipo_di_affidamento": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__contributi_per_il_mantenimento_figli": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__importo_assegno_per_il_mantenimento_figli": [0,5000],
    "dettagli_figli__contributi_economici__obbligato_al_mantenimento": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__beneficiario_assegno_per_mantenimento_figli": "NON SPECIFICATO",
    "dettagli_figli__contributi_economici__modalita_pagamento_assegno_di_mantenimento_del_coniuge": "NON SPECIFICATO"
}
```

---

**Punti chiave da evidenziare nel prompt:**
- Ogni campo ha valori limitati che **devono essere rispettati rigorosamente**.
- Se una query non menziona un dato specifico, il valore deve essere `"NON SPECIFICATO"`.
- **Nessun altro valore** è accettato se non quelli presenti nel `filters_structure`.
- Rispondi esclusivamente nel formato JSON richiesto, senza aggiungere testo introduttivo o commenti.


QUERY UTENTE:

"""
    prompt +=query
    return prompt



# Funzione per ottenere il risultato
def risultato(doc, model, tokenizer):
        messages=prompt_query(doc)
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=messages)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=5000, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        try:
                ris_json = json.loads(result.replace('```json\n','').replace('```','').strip())
        except:
                ris_json = result
        return ris_json



