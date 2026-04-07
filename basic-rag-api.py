import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
from google import genai
from google.genai import types
import chromadb 
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==========================================
# 1. SETUP E CONFIGURAZIONE
# ==========================================

load_dotenv()

# Inizializziamo i client per l'AI e per il Server API
client = genai.Client()
app = FastAPI()

# Inizializza il database vettoriale ChromaDB (nella cartella locale)
chroma_client = chromadb.PersistentClient(path="./db_personale")
collection = chroma_client.get_or_create_collection(name="manuale_aziendale")

# ==========================================
# 2. MODELLI DATI (Pydantic)
# ==========================================

class RichiestaRiassunto(BaseModel):
    """Struttura attesa per l'endpoint /riassumi"""
    testo: str

# ==========================================
# 3. FUNZIONI DI SUPPORTO (Utility)
# ==========================================

def get_embeddings(texts):
    """
    Trasforma una lista di testi in vettori numerici (embeddings).
    Include gestione errori per problemi di rete o chiavi API.
    """
    if not texts:
        return []

    try:
        # Nota: usiamo un client dedicato per gli embeddings
        emb_client = genai.Client(AI_API_KEY=os.environ.get("AI_API_KEY"))
        
        result = emb_client.models.embed_content(
            model='gemini-embedding-001',
            contents=texts
        )
        
        return [e.values for e in result.embeddings]

    except Exception as e:
        print(f"Errore generazione embeddings: {e}")
        return None

# ==========================================
# 4. ENDPOINTS API (Rotte)
# ==========================================

@app.post("/riassumi")
async def genera_riassunto(dati: RichiestaRiassunto):
    """Riceve un testo e restituisce un riassunto strutturato in JSON"""
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=f"Riassumi questo testo in una sola frase. Restituisci ESCLUSIVAMENTE un oggetto JSON con le chiavi 'argomento' e 'riassunto'. Testo: {dati.testo}",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    
    # Decodifica il JSON ricevuto da Gemini e aggiunge info sul modello
    result = json.loads(response.text)
    result["model"] = "gemini-2.5-flash"    
    
    return result


@app.post("/chiedi")
async def elabora_risposta(domanda: str):
    """Sistema RAG: cerca nel DB e risponde usando il contesto trovato"""

    # A. Trasformiamo la domanda in vettori
    embedding = get_embeddings([domanda])

    if embedding is None:
        return {"error": "Impossibile elaborare la domanda al momento."}

    # B. Ricerca semantica nel database ChromaDB (prendiamo i 2 frammenti più vicini)
    results = collection.query( 
        query_embeddings=embedding,
        n_results=2
    )       


    # C. Uniamo i testi trovati per creare il "contesto" da dare all'AI
    contesto_recuperato = " ".join(results['documents'][0])

    # D. Costruiamo il prompt finale con la tecnica del 'Grounding'
    prompt_finale = f"""
        Sei un assistente AI aziendale. Rispondi usando solo il contesto fornito.

        <context>
            {contesto_recuperato}
        </context>

        Domanda: {domanda}
    """  
    
    elenco_fonti = [m["source"] for m in results['metadatas'][0]]


    # E. Chiediamo a Gemini di generare la risposta basata sul contesto
    gemini_res = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=prompt_finale,
                        config=types.GenerateContentConfig(
                                response_mime_type="text/plain",
                        )
                )
                
    
    # F. Prepariamo l'oggetto finale da restituire all'utente
    final_res = {
        "question": domanda,
        "answer": gemini_res.text,
        "sources": list(set(elenco_fonti))
    }

    return final_res


# Configuriamo lo splitter (fuori dalla rotta per efficienza)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Lunghezza massima di ogni pezzo (caratteri)
    chunk_overlap=100,  # Quanti caratteri "copiare" dal pezzo precedente
    separators=["\n\n", "\n", " ", ""]
)

@app.post("/upload")
async def carica_documento(file: UploadFile = File(...)):
    # 1. Leggiamo il file caricato (assumiamo sia un file testuale)
    contenuto_bytes = await file.read()
    testo_completo = contenuto_bytes.decode("utf-8")
    

    # 2. Qui avviene la magia: lo splitter crea una lista di stringhe intelligenti
    chunks = text_splitter.split_text(testo_completo)
    

    metadata_list = [{"source": file.filename} for _ in chunks]
    if not chunks:
        return {"error": "Il file è vuoto o non leggibile."}

    # 3. Generiamo gli embeddings per tutti i chunk
    embeddings = get_embeddings(chunks)
    
    if embeddings is None:
        return {"error": "Errore durante la creazione dei vettori."}

    # 4. Creiamo ID univoci per ogni frammento
    ids = [str(uuid.uuid4()) for _ in chunks]

    # 5. Salviamo tutto su ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadata_list
    )

    return {
        "messaggio": f"File '{file.filename}' elaborato con successo!",
        "pezzi_creati": len(chunks)
    }