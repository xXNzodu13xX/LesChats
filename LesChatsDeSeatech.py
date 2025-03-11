import os
import re
import json
import pickle
import time
import uuid
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session

# Importations ML et API
try:
    import torch
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_IMPORTS_SUCCESS = True
except ImportError:
    ML_IMPORTS_SUCCESS = False

try:
    from groq import Groq
    GROQ_IMPORT_SUCCESS = True
except ImportError:
    GROQ_IMPORT_SUCCESS = False

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seatech_chatbot")

# Répertoires et fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DB_DIR = os.path.join(BASE_DIR, "database", "user_database")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
for directory in [DATA_DIR, USER_DB_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "chunk_embeddings.pkl")
CHUNKS_CACHE = os.path.join(CACHE_DIR, "chunks_with_sources.pkl")
QA_STORAGE = os.path.join(CACHE_DIR, "user_qa_memory.json")

# ===== CONFIGURATION =====
GROQ_API_KEY = "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"
LLM_MODEL = "llama-3.3-70b-versatile"
CONFIDENCE_THRESHOLD = 0.93

# Acronymes utilisés
ACRONYMS = {
    "SN": "Systèmes Numériques",
    "MTX": "Matériaux",
    "APP": "Apprentissage",
    "FISE": "Formation Initiale Sous Statut d'Étudiant",
    "FISA": "Formation Initiale Sous Statut d'Apprenti",
    "UTLN": "Université de Toulon"
}

# Pour l'exemple, nous définissons CONTACTS vide (à compléter selon vos besoins)
CONTACTS = {}

# ===== INITIALISATION DES CLIENTS =====
if GROQ_IMPORT_SUCCESS:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Client GROQ initialisé")
    except Exception as e:
        logger.error(f"Erreur initialisation client GROQ: {e}")
        groq_client = None
else:
    groq_client = None
    logger.warning("GROQ non disponible - vérifiez l'installation")

if ML_IMPORTS_SUCCESS:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = SentenceTransformer("intfloat/e5-large-v2", device=device)
        logger.info(f"Modèle d'embedding chargé sur {device}")
    except Exception as e:
        logger.error(f"Erreur chargement modèle d'embedding: {e}")
        embedding_model = None
else:
    embedding_model = None
    logger.warning("Bibliothèques ML non disponibles")

# ===== FONCTIONS UTILITAIRES =====
def expand_acronyms_in_query(query):
    """Étend les acronymes dans la requête avec une meilleure détection."""
    expanded_query = query
    for acronym, expansion in ACRONYMS.items():
        # Détection plus précise avec une expression régulière
        pattern = r'\b' + re.escape(acronym) + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            expanded_query = re.sub(pattern, f"{acronym} ({expansion})", expanded_query, flags=re.IGNORECASE)
            logger.info(f"Acronyme détecté et étendu: {acronym} -> {expansion}")
    
    # Si des acronymes ont été étendus, on ajoute une note
    if expanded_query != query:
        expanded_query += " " + " ".join([expansion for acronym, expansion in ACRONYMS.items() 
                                         if re.search(r'\b' + re.escape(acronym) + r'\b', query, re.IGNORECASE)])
    
    return expanded_query

def convert_markdown_to_html(text):
    """
    Conversion améliorée du Markdown vers HTML avec prise en charge de plus de formats.
    """
    # Gestion des titres (#, ##, etc.)
    def replace_heading(match):
        hashes = match.group(1)
        level = len(hashes)
        title = match.group(2).strip()
        return f"<h{level}>{title}</h{level}>"
    
    text = re.sub(r'^(#{1,6})\s+(.*)$', replace_heading, text, flags=re.MULTILINE)
    
    # Conversion des formats gras et italique
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)  # Alternative pour le gras
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)  # Alternative pour l'italique
    
    # Gestion des liens [texte](url)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', text)
    
    # Traitement des listes à puces et numérotées
    lines = text.splitlines()
    html_lines = []
    in_ul = False
    in_ol = False
    
    for line in lines:
        # Liste à puces
        if line.strip().startswith("* ") or line.strip().startswith("- "):
            if not in_ul:
                if in_ol:
                    html_lines.append("</ol>")
                    in_ol = False
                html_lines.append("<ul>")
                in_ul = True
            content = re.sub(r'^\s*[\*\-]\s+(.*)', r'\1', line)
            html_lines.append(f"<li>{content}</li>")
        
        # Liste numérotée
        elif re.match(r'^\s*\d+\.\s+', line):
            if not in_ol:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                html_lines.append("<ol>")
                in_ol = True
            content = re.sub(r'^\s*\d+\.\s+(.*)', r'\1', line)
            html_lines.append(f"<li>{content}</li>")
        
        # Ligne normale
        else:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append(line)
    
    # Fermer les listes si nécessaire
    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    
    text = "\n".join(html_lines)
    
    # Découpage en paragraphes pour les blocs de texte non déjà formatés
    paragraphs = []
    for block in re.split(r'\n\s*\n', text):
        block = block.strip()
        # Vérifier si le bloc contient déjà des balises HTML
        if not re.match(r'^<\/?(h\d|ul|ol|li|blockquote|pre|table)', block):
            if block:  # Ne pas ajouter de paragraphe vide
                block = f"<p>{block}</p>"
        paragraphs.append(block)
    
    return "\n".join(paragraphs)

def generate_basic_answer(query, context, found_info):
    """Fallback basique en l'absence du client GROQ."""
    answer = "<p>Désolé, le service de génération de réponse n'est pas disponible actuellement. Veuillez réessayer plus tard ou contacter l'administrateur.</p>"
    return answer

# ===== GESTION DES DONNÉES =====
def create_default_data():
    """Crée des fichiers de données par défaut si DATA_DIR est vide."""
    if not os.listdir(DATA_DIR):
        # Création d'un fichier d'acronymes
        with open(os.path.join(DATA_DIR, "acronymes.txt"), "w", encoding="utf-8") as f:
            f.write("# Acronymes utilisés à SeaTech\n\n")
            for acronym, meaning in ACRONYMS.items():
                f.write(f"{acronym}: {meaning}\n")
        # Fichier de contacts (seulement si CONTACTS est renseigné)
        with open(os.path.join(DATA_DIR, "contacts.txt"), "w", encoding="utf-8") as f:
            f.write("# Contacts importants à SeaTech\n\n")
            if CONTACTS:
                for name, info in CONTACTS.items():
                    f.write(f"{name}: {info.get('role', 'N/A')} - {info.get('email', 'N/A')}\n")
            else:
                f.write("Aucun contact défini.\n")
        # Fichier d'information générale
        with open(os.path.join(DATA_DIR, "info_generale.json"), "w", encoding="utf-8") as f:
            f.write("# SeaTech - École d'ingénieurs\n\n")
            f.write("SeaTech est une école d'ingénieurs de l'Université de Toulon (UTLN). ")
            f.write("Elle propose plusieurs formations d'ingénieur dont les spécialités SN (Systèmes Numériques) ")
            f.write("et MTX (Matériaux). Les formations peuvent être suivies en statut étudiant (FISE) ou en apprentissage (FISA).\n")

def load_data():
    """Charge les fichiers dans DATA_DIR et découpe le contenu en chunks avec timestamp de dernière modification."""
    create_default_data()
    chunks_with_sources = []
    valid_extensions = ['.txt', '.md', '.html', '.csv','.json']
    
    # Charger un cache des timestamps si disponible
    timestamp_cache_path = os.path.join(CACHE_DIR, "file_timestamps.json")
    file_timestamps = {}
    reload_all = False
    
    if os.path.exists(timestamp_cache_path):
        try:
            with open(timestamp_cache_path, "r", encoding="utf-8") as f:
                file_timestamps = json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement cache timestamps: {e}")
            reload_all = True
    
    # Vérifier et charger les fichiers
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in valid_extensions):
            # Vérifier si le fichier a été modifié
            mtime = os.path.getmtime(file_path)
            
            if reload_all or filename not in file_timestamps or file_timestamps[filename] < mtime:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Découpage en chunks selon les paragraphes
                    raw_chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
                    for chunk in raw_chunks:
                        chunks_with_sources.append((chunk, filename))
                    logger.info(f"Fichier {filename} chargé : {len(raw_chunks)} chunks")
                    file_timestamps[filename] = mtime
                except Exception as e:
                    logger.error(f"Erreur chargement {filename}: {e}")
    
    # Sauvegarder les timestamps mis à jour
    try:
        with open(timestamp_cache_path, "w", encoding="utf-8") as f:
            json.dump(file_timestamps, f)
    except Exception as e:
        logger.error(f"Erreur sauvegarde cache timestamps: {e}")
    
    return chunks_with_sources

def compute_embeddings(chunks_with_sources):
    """Calcule ou charge les embeddings pour les chunks."""
    if not embedding_model:
        logger.warning("Modèle d'embedding non disponible")
        dummy_embeddings = np.zeros((len(chunks_with_sources), 384))
        return dummy_embeddings, chunks_with_sources
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNKS_CACHE):
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                cached_embeddings = pickle.load(f)
            with open(CHUNKS_CACHE, "rb") as f:
                cached_chunks = pickle.load(f)
            if len(cached_embeddings) == len(cached_chunks):
                logger.info(f"Cache chargé : {len(cached_embeddings)} embeddings")
                return cached_embeddings, cached_chunks
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
    try:
        chunk_texts = [chunk[0] for chunk in chunks_with_sources]
        embeddings = embedding_model.encode(chunk_texts, batch_size=8, convert_to_tensor=True)
        chunk_embeddings = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else np.array(embeddings)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(chunk_embeddings, f)
        with open(CHUNKS_CACHE, "wb") as f:
            pickle.dump(chunks_with_sources, f)
        logger.info(f"Embeddings calculés et sauvegardés : {len(chunk_embeddings)}")
        return chunk_embeddings, chunks_with_sources
    except Exception as e:
        logger.error(f"Erreur calcul embeddings: {e}")
        dummy_embeddings = np.zeros((len(chunks_with_sources), 384))
        return dummy_embeddings, chunks_with_sources

# ===== INDEXATION AVEC FAISS =====
def setup_search_index(embeddings):
    """
    Configure l'index FAISS.
    Utilise IndexIVFFlat pour un grand nombre d'embeddings, sinon IndexFlatL2.
    """
    if not ML_IMPORTS_SUCCESS or embeddings.size == 0:
        logger.warning("Index de recherche non disponible")
        return None, False
    try:
        d = embeddings.shape[1]
        if embeddings.shape[0] > 1000:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
            index.train(embeddings)
            index.add(embeddings)
            logger.info(f"Index FAISS IVFFlat créé avec {index.ntotal} vecteurs")
            return index, True
        else:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            logger.info(f"Index FAISS FlatL2 créé avec {index.ntotal} vecteurs")
            return index, True
    except Exception as e:
        logger.error(f"Erreur initialisation index: {e}")
        return None, False

def search_similar_chunks(query, index, is_faiss, embeddings, chunks_data, top_n=5):
    """Recherche les chunks les plus similaires à la requête."""
    if not embedding_model or not index:
        return keyword_search(query, chunks_data, top_n)
    try:
        query_expanded = expand_acronyms_in_query(query)
        query_embedding = embedding_model.encode(query_expanded, convert_to_tensor=False)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        if is_faiss:
            distances, indices = index.search(query_embedding, min(top_n * 2, len(chunks_data)))
            distances, indices = distances[0], indices[0]
            results = []
            for i, idx in enumerate(indices):
                if 0 <= idx < len(chunks_data):
                    score = 1 / (1 + distances[i])
                    results.append((chunks_data[idx][0], chunks_data[idx][1], score))
        else:
            similarities = cosine_similarity(query_embedding, embeddings)[0]
            top_indices = np.argsort(similarities)[-top_n*2:][::-1]
            results = [(chunks_data[idx][0], chunks_data[idx][1], similarities[idx])
                       for idx in top_indices if similarities[idx] > 0.1]
        results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n]
        return results
    except Exception as e:
        logger.error(f"Erreur de recherche: {e}")
        return keyword_search(query, chunks_data, top_n)

def keyword_search(query, chunks_data, top_n=5):
    """Recherche par mots-clés en fallback."""
    query_terms = query.lower().split()
    results = []
    # Recherche d'acronymes
    for acronym in ACRONYMS:
        if acronym.lower() in query.lower():
            for chunk, source in chunks_data:
                if acronym in chunk:
                    results.append((chunk, source, 0.85))
    for chunk, source in chunks_data:
        score = sum(0.1 for term in query_terms if term in chunk.lower())
        if score > 0:
            results.append((chunk, source, score))
    if not results:
        results.append(("Aucune information spécifique trouvée.", "fallback.txt", 0.1))
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

# ===== FORMATAGE DES SOURCES ET LOGS =====
def format_sources(results, for_freddy=False):
    """Formate les résultats en HTML pour l'affichage des sources."""
    html = '<div class="sources-container">'
    for text, source, score in results:
        relevance_class = "high-relevance" if score > 0.8 else "medium-relevance" if score > 0.6 else "low-relevance"
        if for_freddy:
            preview = text[:200] + ("..." if len(text) > 200 else "")
            html += f'''
            <div class="freddy-source-block" onclick="this.classList.toggle('expanded')">
                <div class="freddy-source-header">
                    <span class="source-name">{source}</span>
                    <span class="{relevance_class}">{score:.2f}</span>
                </div>
                <div class="freddy-source-content">{preview}</div>
            </div>
            '''
        else:
            html += f'''
            <div class="source-block">
                <div class="source-header">
                    <span class="source-name">{source}</span>
                    <span class="relevance-score">{score:.2f}</span>
                </div>
                <div class="source-content">{text}</div>
            </div>
            '''
    html += '</div>'
    return html

def create_freddy_logs(query, results):
    """Crée des logs HTML détaillés pour le module Freddy."""
    current_time = datetime.now().strftime("%H:%M:%S")
    high_relevance = sum(1 for _, _, score in results if score > 0.8)
    medium_relevance = sum(1 for _, _, score in results if 0.6 < score <= 0.8)
    low_relevance = sum(1 for _, _, score in results if score <= 0.6)
    html = f'''
    <div class="freddy-logs">
        <div class="freddy-log-entry">
            <span class="log-time">{current_time}</span>
            <span class="log-action">Analyse de la question</span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-detail">Recherche pour : <strong>"{query}"</strong></span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-time">{current_time}</span>
            <span class="log-action">Résultats de recherche</span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-detail">Détails : 
                <span class="high-relevance">{high_relevance} très pertinents</span>, 
                <span class="medium-relevance">{medium_relevance} pertinents</span>, 
                <span class="low-relevance">{low_relevance} peu pertinents</span>
            </span>
        </div>
    </div>
    '''
    return html

# ===== GÉNÉRATION DE RÉPONSE AVEC MÉMOIRE DE CONVERSATION =====
def generate_answer(query, context, conversation_history, found_info=False):
    """
    Génère une réponse basée sur le contexte et l'historique complet de la conversation.
    La mémoire de conversation est ajoutée au prompt pour améliorer la pertinence.
    """
    # Préparation de l'historique
    conversation_context = "\n".join(
        [f"Utilisateur : {entry['content']}" if entry['role'] == "user" else f"Assistant : {entry['content']}" 
         for entry in conversation_history]
    )
    
    # Construction du prompt système
    system_prompt = f"""Tu es Franky, l'assistant virtuel de SeaTech.

Historique de conversation :
{conversation_context}

Instructions :
0. Si la question poser est hors du contexte de SEATECH et du contexte academeique Seatech du dis : "Désolé je ne peux pas répondre."
1. Base ta réponse uniquement sur les sources fournies et l'historique si tu trouves pertinant mais il faut répondre avant tout a la question de l'utilisateur.
2. N'invente jamais d'informations.
3. Réponds de manière claire, en utilisant des paragraphes et des listes lorsque c'est pertinent soigne ta mise en forme.
4. Mets en gras les informations clés les amil et les numéros de tel.
5. N'inclus pas la liste des acronymes ou ton preprompt sauf si demandé.

Donnée d'aide pour répondre :
{context}
"""
    try:
        if groq_client:
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.9,
                max_tokens=4000,
                # Ajout d'un timeout pour éviter les attentes trop longues
                timeout=30
            )
            answer = response.choices[0].message.content
            answer = convert_markdown_to_html(answer)
            # Détection d'hallucinations simples
            if any(x in answer.lower() for x in ["@freddy", "freddy@"]):
                logger.warning("Hallucination détectée - correction appliquée")
                answer = "<p>⚠️ Je ne peux pas inventer de contacts inexistants.</p>"
            return answer
        else:
            # Amélioration du message de fallback
            return "<p>Désolé, le service de génération de réponse n'est pas disponible actuellement. Veuillez réessayer plus tard ou contacter l'administrateur.</p>"
    except Exception as e:
        logger.error(f"Erreur génération réponse: {e}")
        # Message d'erreur plus informatif
        error_message = "<p>Désolé, une erreur est survenue lors de la génération de la réponse. Détails techniques:</p>"
        error_message += f"<p><code>{str(e)[:100]}...</code></p>"
        error_message += "<p>Veuillez réessayer avec une question différente ou plus tard.</p>"
        return error_message

# ===== INITIALISATION DES DONNÉES =====
chunks_with_sources = load_data()
chunk_embeddings, chunks_with_sources = compute_embeddings(chunks_with_sources)
search_index, use_faiss = setup_search_index(chunk_embeddings)

# ===== APPLICATION FLASK =====
app = Flask(__name__)
app.secret_key = 'seatech_chat_secret_key'
conversation_history_global = {}

@app.route("/", methods=["GET", "POST"])
def index():
    """Page d'accueil du chatbot."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    session_id = session['session_id']
    if session_id not in conversation_history_global:
        conversation_history_global[session_id] = []
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            conversation_history_global[session_id].append({"role": "user", "content": user_query})
            start_time = time.time()
            results = search_similar_chunks(user_query, search_index, use_faiss, chunk_embeddings, chunks_with_sources)
            # Contexte des chunks trouvés
            context_chunks = "\n\n".join([text for text, _, _ in results])
            found_info = any(score > CONFIDENCE_THRESHOLD for _, _, score in results)
            answer = generate_answer(user_query, context_chunks, conversation_history_global[session_id], found_info)
            processing_time = time.time() - start_time
            sources_html = format_sources(results)
            freddy_html = create_freddy_logs(user_query, results) + format_sources(results, for_freddy=True)
            conversation_history_global[session_id].append({
                "role": "assistant", 
                "content": answer,
                "sources": sources_html,
                "freddy_logs": freddy_html,
                "processing_time": f"{processing_time:.2f}s"
            })
    conv = conversation_history_global.get(session.get('session_id'), [])
    current_datetime = datetime.now()
    return render_template("index.html", conversation=conv, query="", current_datetime=current_datetime)

@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Endpoint API pour la recherche."""
    start_time = time.time()
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"error": "Question vide"}), 400
        
        # Gestion de la session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        session_id = session['session_id']
        if session_id not in conversation_history_global:
            conversation_history_global[session_id] = []
            
        # Ajout à l'historique de conversation
        conversation_history_global[session_id].append({"role": "user", "content": user_query})
        
        # Recherche d'informations
        try:
            results = search_similar_chunks(user_query, search_index, use_faiss, chunk_embeddings, chunks_with_sources)
            context_chunks = "\n\n".join([text for text, _, _ in results])
            found_info = any(score > CONFIDENCE_THRESHOLD for _, _, score in results)
        except Exception as search_error:
            logger.error(f"Erreur de recherche: {search_error}")
            results = [("Une erreur s'est produite lors de la recherche.", "error.txt", 0.1)]
            context_chunks = "Informations non disponibles en raison d'une erreur."
            found_info = False
        
        # Génération de réponse
        answer = generate_answer(user_query, context_chunks, conversation_history_global[session_id], found_info)
        
        # Formatage des sources et logs
        sources_html = format_sources(results)
        freddy_html = create_freddy_logs(user_query, results) + format_sources(results, for_freddy=True)
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Mise à jour de l'historique de conversation
        conversation_history_global[session_id].append({
            "role": "assistant", 
            "content": answer,
            "sources": sources_html,
            "freddy_logs": freddy_html,
            "processing_time": f"{processing_time:.2f}s"
        })
        
        # Réponse de l'API
        return jsonify({
            "response": answer,
            "freddy_logs": freddy_html,
            "sources": sources_html,
            "sources_found": found_info,
            "processing_time": f"{processing_time:.2f}s",
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Erreur API globale: {e}")
        processing_time = time.time() - start_time
        
        # Réponse d'erreur plus détaillée
        return jsonify({
            "response": f"<p>Désolé, une erreur s'est produite: {str(e)[:50]}...</p><p>Veuillez réessayer.</p>",
            "freddy_logs": f"<div class='freddy-logs'><div class='freddy-log-entry'><span class='log-action'>Erreur</span><span class='log-detail'>{str(e)[:100]}...</span></div></div>",
            "sources_found": False,
            "processing_time": f"{processing_time:.2f}s",
            "status": "error",
            "error_type": str(type(e).__name__)
        }), 500

@app.route('/static/<path:path>')
def send_static(path):
    """Fournit les fichiers statiques."""
    return send_from_directory('static', path)

if __name__ == "__main__":
    # Création des répertoires statiques si nécessaires
    if not os.path.exists("static"):
        os.makedirs("static", exist_ok=True)
        os.makedirs("static/css", exist_ok=True)
        os.makedirs("static/img", exist_ok=True)
    templates_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)