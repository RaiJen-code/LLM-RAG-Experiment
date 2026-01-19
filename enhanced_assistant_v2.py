"""
Enhanced Voice Assistant v2.0
Designed for LLM+RAG vs LLM Standalone Performance Comparison
Platform: Jetson Orin Nano 8GB - Fully Offline

Author: aRJey
Date: 2025
"""

import whisper
import requests
import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
import faiss
from sentence_transformers import SentenceTransformer
import torch
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import glob

# ============================================================================
# DISABLE ONLINE CHECKS - FULLY OFFLINE MODE
# ============================================================================
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'transformers')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Centralized configuration for the voice assistant"""
    
    # Device configuration
    DEVICE = "cpu"  # CPU mode for stability on Jetson
    
    # Model configurations
    WHISPER_MODEL = "tiny"  # Fast ASR model
    EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    # Ollama LLM configuration
    OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:3b"
    
    # Conversation settings
    MAX_HISTORY = 5
    MAX_TOKENS = 150
    
    # Audio settings
    AUDIO_DURATION = 5  # seconds
    SAMPLE_RATE = 16000
    AUDIO_DEVICE = 25  # C270 HD WEBCAM
    
    # RAG settings
    RAG_TOP_K = 3
    RAG_SIMILARITY_THRESHOLD = 0.3
    RAG_CHUNK_SIZE = 500
    RAG_CHUNK_OVERLAP = 50
    
    # Paths
    PROJECT_DIR = Path(__file__).parent
    KB_DIR = PROJECT_DIR / "knowledge_base"
    KB_DEFAULT_DIR = KB_DIR / "default_knowledge"
    KB_USER_DIR = KB_DIR / "user_documents"
    RESULTS_DIR = PROJECT_DIR / "experiment_results"
    LOG_DIR = PROJECT_DIR / "logs"
    
    # Audio files
    BIP_SOUND = PROJECT_DIR / "assets" / "bip.wav"
    BIP2_SOUND = PROJECT_DIR / "assets" / "bip2.wav"
    
    # Piper TTS
    PIPER_BIN = "/home/rangga/piper/build/piper"
    PIPER_MODEL = "/usr/local/share/piper/models/en_US-lessac-medium.onnx"
    
    # Testing configuration
    NUM_REPETITIONS = 5  # Number of repetitions per question
    
    # System prompt
    SYSTEM_PROMPT = """You are a knowledgeable AI assistant specialized in mechatronics, 
electronics, microcontrollers, and embedded systems. Provide accurate, concise answers.
Keep responses under 2-3 sentences for voice interaction. Be precise and technical."""
    
    @classmethod
    def setup_directories(cls):
        """Create necessary directories"""
        for directory in [cls.KB_DIR, cls.KB_DEFAULT_DIR, cls.KB_USER_DIR, 
                         cls.RESULTS_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# Setup directories
Config.setup_directories()

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging system"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = Config.LOG_DIR / f'assistant_{timestamp}.log'
    
    logger = logging.getLogger('VoiceAssistant')
    logger.setLevel(logging.DEBUG)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

logger = setup_logging()

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

class PerformanceTracker:
    """Track and log performance metrics"""
    
    def __init__(self):
        self.metrics = []
        self.current_query = {}
        self.session_start = datetime.now()
    
    def start_query(self, query_text: str, mode: str):
        """Start tracking a new query"""
        self.current_query = {
            'query': query_text,
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
            'timings': {},
            'errors': []
        }
    
    def log_timing(self, component: str, duration: float):
        """Log timing for a component"""
        self.current_query['timings'][component] = round(duration, 4)
        logger.debug(f"{component}: {duration:.3f}s")
    
    def log_error(self, error: str):
        """Log an error"""
        self.current_query['errors'].append(error)
        logger.error(error)
    
    def log_result(self, response: str, rag_info: Dict = None):
        """Log the final result"""
        self.current_query['response'] = response
        self.current_query['success'] = len(self.current_query['errors']) == 0
        
        if rag_info:
            self.current_query['rag_info'] = {
                'docs_retrieved': rag_info.get('docs_retrieved', 0),
                'avg_similarity': rag_info.get('avg_similarity', 0)
            }
        
        total_time = sum(self.current_query['timings'].values())
        self.current_query['total_time'] = round(total_time, 4)
        
        # Add to metrics
        self.metrics.append(self.current_query.copy())
    
    def save_metrics(self, filename: str = None):
        """Save metrics to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = Config.RESULTS_DIR / f'metrics_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump({
                'session_info': {
                    'start_time': self.session_start.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_queries': len(self.metrics)
                },
                'metrics': self.metrics
            }, f, indent=2)
        
        logger.info(f"Metrics saved to {filename}")
        return filename

# Global performance tracker
perf_tracker = PerformanceTracker()

# ============================================================================
# ENHANCED VECTOR DATABASE FOR RAG
# ============================================================================

class KnowledgeBase:
    """Enhanced knowledge base with PDF processing and hybrid sources"""
    
    def __init__(self, embedding_model: SentenceTransformer, dim: int = 384):
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []
        self.metadata = []
        self.embedding_model = embedding_model
        self.dim = dim
        logger.info("Knowledge Base initialized")
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                if last_period > chunk_size * 0.5:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_documents(self, docs: List[str], source: str = "manual"):
        """Add documents with chunking and metadata"""
        added_count = 0
        
        for doc_idx, doc in enumerate(docs):
            if not doc or len(doc.strip()) < 20:
                continue
                
            chunks = self.chunk_text(doc, Config.RAG_CHUNK_SIZE, Config.RAG_CHUNK_OVERLAP)
            
            for chunk_idx, chunk in enumerate(chunks):
                if len(chunk.strip()) < 30:  # Skip very short chunks
                    continue
                
                metadata = {
                    'source': source,
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'text': chunk
                }
                
                self.documents.append(chunk)
                self.metadata.append(metadata)
                added_count += 1
        
        # Rebuild index
        if self.documents:
            embeddings = self.embedding_model.encode(self.documents)
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.array(embeddings, dtype=np.float32))
        
        logger.info(f"Added {added_count} chunks from '{source}'")
        return added_count
    
    def load_from_pdf(self, pdf_path: Path):
        """Load and process PDF document"""
        try:
            import PyPDF2
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                    except Exception as e:
                        logger.warning(f"Could not extract page {page_num} from {pdf_path.name}: {e}")
                
                if full_text.strip():
                    self.add_documents([full_text], source=pdf_path.name)
                    logger.info(f"Successfully loaded PDF: {pdf_path.name}")
                    return True
                else:
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            return False
    
    def load_from_text(self, text_path: Path):
        """Load from text file"""
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if content.strip():
                self.add_documents([content], source=text_path.name)
                logger.info(f"Successfully loaded text file: {text_path.name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error loading text file {text_path}: {e}")
            return False
    
    def load_from_json(self, json_path: Path):
        """Load from JSON file (list of facts/documents)"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                docs = [str(item) for item in data if item]
                self.add_documents(docs, source=json_path.name)
            elif isinstance(data, dict) and 'documents' in data:
                self.add_documents(data['documents'], source=json_path.name)
            
            logger.info(f"Successfully loaded JSON: {json_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading JSON {json_path}: {e}")
            return False
    
    def load_all_documents(self):
        """Load all documents from knowledge base directories"""
        total_loaded = 0
        
        # Load default knowledge
        if Config.KB_DEFAULT_DIR.exists():
            logger.info(f"Loading default knowledge from: {Config.KB_DEFAULT_DIR}")
            for pattern in ['*.txt', '*.json', '*.pdf']:
                for file_path in Config.KB_DEFAULT_DIR.glob(pattern):
                    if file_path.suffix == '.pdf':
                        if self.load_from_pdf(file_path):
                            total_loaded += 1
                    elif file_path.suffix == '.json':
                        if self.load_from_json(file_path):
                            total_loaded += 1
                    elif file_path.suffix == '.txt':
                        if self.load_from_text(file_path):
                            total_loaded += 1
        
        # Load user documents
        if Config.KB_USER_DIR.exists():
            logger.info(f"Loading user documents from: {Config.KB_USER_DIR}")
            for pattern in ['*.pdf', '*.txt']:
                for file_path in Config.KB_USER_DIR.glob(pattern):
                    if file_path.suffix == '.pdf':
                        if self.load_from_pdf(file_path):
                            total_loaded += 1
                    elif file_path.suffix == '.txt':
                        if self.load_from_text(file_path):
                            total_loaded += 1
        
        logger.info(f"Total files loaded: {total_loaded}")
        return total_loaded
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search with similarity scores and metadata"""
        if not self.documents:
            logger.warning("No documents in knowledge base")
            return []
        
        query_embedding = self.embedding_model.encode([query])[0].astype(np.float32)
        
        distances, indices = self.index.search(
            np.array([query_embedding]), 
            min(top_k, len(self.documents))
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + dist)
            
            if similarity >= Config.RAG_SIMILARITY_THRESHOLD:
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarity),
                    'distance': float(dist)
                })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics"""
        sources = {}
        for meta in self.metadata:
            source = meta['source']
            sources[source] = sources.get(source, 0) + 1
        
        return {
            'total_chunks': len(self.documents),
            'total_chars': sum(len(doc) for doc in self.documents),
            'sources': sources,
            'num_sources': len(sources)
        }

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

logger.info("="*60)
logger.info("Initializing Models...")
logger.info("="*60)

# Initialize embedding model
embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)
logger.info(f"✓ Embedding model loaded: {Config.EMBEDDING_MODEL}")

# Initialize Whisper ASR
whisper_model = whisper.load_model(Config.WHISPER_MODEL, device=Config.DEVICE)
logger.info(f"✓ Whisper model loaded: {Config.WHISPER_MODEL}")

# Initialize Knowledge Base
kb = KnowledgeBase(embedding_model, dim=Config.EMBEDDING_DIM)

# Load all documents from knowledge base
num_files = kb.load_all_documents()

if num_files == 0:
    logger.warning("No knowledge base files found!")
    logger.warning(f"Please add documents to:")
    logger.warning(f"  - Default: {Config.KB_DEFAULT_DIR}")
    logger.warning(f"  - User docs: {Config.KB_USER_DIR}")
else:
    stats = kb.get_stats()
    logger.info("Knowledge Base Statistics:")
    logger.info(f"  Total chunks: {stats['total_chunks']}")
    logger.info(f"  Total sources: {stats['num_sources']}")
    for source, count in stats['sources'].items():
        logger.info(f"    - {source}: {count} chunks")

# Conversation history
conversation_history = []

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_to_history(user_msg: str, assistant_msg: str):
    """Add exchange to conversation history"""
    conversation_history.append({
        "user": user_msg,
        "assistant": assistant_msg
    })
    
    if len(conversation_history) > Config.MAX_HISTORY:
        conversation_history.pop(0)

def get_history_context() -> str:
    """Get formatted conversation history"""
    if not conversation_history:
        return ""
    
    history_text = "Recent conversation:\n"
    for exchange in conversation_history[-3:]:
        history_text += f"User: {exchange['user']}\n"
        history_text += f"Assistant: {exchange['assistant']}\n"
    
    return history_text

# ============================================================================
# AUDIO FUNCTIONS
# ============================================================================

def play_sound(sound_file: Path):
    """Play audio notification"""
    try:
        if sound_file.exists():
            os.system(f"aplay -D plughw:0,0 {sound_file} 2>/dev/null")
    except Exception as e:
        logger.debug(f"Could not play sound: {e}")

def record_audio(filename: str, duration: int = None) -> float:
    """Record audio and return recording time"""
    if duration is None:
        duration = Config.AUDIO_DURATION
    
    start_time = time.time()
    play_sound(Config.BIP_SOUND)
    
    try:
        audio = sd.rec(
            int(duration * Config.SAMPLE_RATE), 
            samplerate=Config.SAMPLE_RATE, 
            channels=1, 
            dtype='int16', 
            device=Config.AUDIO_DEVICE
        )
        sd.wait()
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(Config.SAMPLE_RATE)
            wf.writeframes(audio.tobytes())
        
        play_sound(Config.BIP2_SOUND)
        return time.time() - start_time
        
    except Exception as e:
        logger.error(f"Recording error: {e}")
        raise

def transcribe_audio(filename: str) -> Tuple[str, float]:
    """Transcribe audio file"""
    start_time = time.time()
    
    try:
        result = whisper_model.transcribe(filename, language="en", fp16=False)
        transcription = result['text'].strip()
        transcription_time = time.time() - start_time
        
        return transcription, transcription_time
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise

def text_to_speech(text: str) -> float:
    """Convert text to speech"""
    start_time = time.time()
    response_file = Config.PROJECT_DIR / "response.wav"
    
    try:
        safe_text = text.replace('"', '\\"').replace("'", "\\'")
        cmd = f'echo "{safe_text}" | {Config.PIPER_BIN} --model {Config.PIPER_MODEL} --output_file {response_file} && aplay -D plughw:0,0 {response_file} 2>/dev/null'
        os.system(cmd)
        
        return time.time() - start_time
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return 0.0

# ============================================================================
# LLM INTERACTION
# ============================================================================

def check_ollama_server() -> bool:
    """Check if Ollama server is running"""
    try:
        response = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            logger.info(f"✓ Ollama server running ({len(models)} models)")
            
            model_names = [m['name'] for m in models]
            if Config.OLLAMA_MODEL in model_names:
                logger.info(f"✓ Using model: {Config.OLLAMA_MODEL}")
            else:
                logger.warning(f"⚠ Model {Config.OLLAMA_MODEL} not found")
            
            return True
    except Exception as e:
        logger.error(f"✗ Ollama server not running: {e}")
        return False

def query_llm(query: str, context: str = "") -> Tuple[str, float]:
    """Query LLM with optional context"""
    start_time = time.time()
    
    history_text = get_history_context()
    context_text = f"Context: {context}\n\n" if context else ""
    
    full_prompt = f"""{Config.SYSTEM_PROMPT}

{history_text}{context_text}User: {query}
Assistant:"""
    
    data = {
        "model": Config.OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": Config.MAX_TOKENS,
            "top_k": 40,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            Config.OLLAMA_URL,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json().get('response', '').strip()
            inference_time = time.time() - start_time
            
            add_to_history(query, result)
            return result, inference_time
        else:
            error_msg = f"Ollama error (status {response.status_code})"
            logger.error(error_msg)
            return error_msg, 0.0
            
    except Exception as e:
        error_msg = f"LLM error: {str(e)[:50]}"
        logger.error(error_msg)
        return error_msg, 0.0

def query_with_rag(query: str) -> Tuple[str, float, Dict]:
    """Query LLM with RAG"""
    # Retrieve relevant documents
    retrieved_docs = kb.search(query, top_k=Config.RAG_TOP_K)
    
    rag_info = {
        'docs_retrieved': len(retrieved_docs),
        'docs': retrieved_docs,
        'avg_similarity': 0
    }
    
    context = ""
    if retrieved_docs:
        context = "Relevant information:\n"
        similarities = []
        
        for i, doc_info in enumerate(retrieved_docs, 1):
            context += f"{i}. {doc_info['text']}\n"
            similarities.append(doc_info['similarity'])
        
        rag_info['avg_similarity'] = sum(similarities) / len(similarities)
        logger.debug(f"Retrieved {len(retrieved_docs)} docs (avg sim: {rag_info['avg_similarity']:.3f})")
    
    # Query LLM with context
    response, llm_time = query_llm(query, context)
    
    return response, llm_time, rag_info

# ============================================================================
# MAIN TESTING FUNCTIONS
# ============================================================================

def process_single_query(query_text: str, use_rag: bool = True) -> Dict:
    """Process a single query with performance tracking"""
    mode = "RAG" if use_rag else "Non-RAG"
    perf_tracker.start_query(query_text, mode)
    
    result = {
        'query': query_text,
        'mode': mode,
        'use_rag': use_rag,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Get LLM response
        if use_rag:
            response, llm_time, rag_info = query_with_rag(query_text)
            result['rag_info'] = rag_info
        else:
            response, llm_time = query_llm(query_text)
            rag_info = None
        
        result['response'] = response
        result['llm_time'] = llm_time
        result['total_time'] = llm_time
        result['success'] = not response.startswith("Error")
        
        perf_tracker.log_timing('llm_inference', llm_time)
        perf_tracker.log_result(response, rag_info)
        
    except Exception as e:
        error_msg = f"Error processing query: {e}"
        logger.error(error_msg)
        perf_tracker.log_error(error_msg)
        result['success'] = False
        result['error'] = str(e)
    
    return result

def run_voice_test_session(questions: List[str], num_repetitions: int = 5):
    """
    Run a complete voice testing session
    This is the main function for real voice testing
    """
    logger.info("="*60)
    logger.info("VOICE TESTING SESSION - STARTING")
    logger.info("="*60)
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"Repetitions: {num_repetitions}")
    logger.info(f"Total tests: {len(questions) * num_repetitions * 2}")
    logger.info("="*60)
    
    if not check_ollama_server():
        logger.error("Ollama server not running!")
        return None
    
    all_results = []
    
    for q_idx, question in enumerate(questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"QUESTION {q_idx}/{len(questions)}: {question}")
        logger.info(f"{'='*60}")
        
        # Test each mode (Non-RAG and RAG)
        for mode in ['Non-RAG', 'RAG']:
            use_rag = (mode == 'RAG')
            logger.info(f"\n--- Mode: {mode} ---")
            
            for rep in range(num_repetitions):
                logger.info(f"\nRepetition {rep+1}/{num_repetitions}")
                
                try:
                    # Record audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                        logger.info("🎤 Recording... (speak now)")
                        rec_time = record_audio(tmpfile.name)
                        
                        # Transcribe
                        transcription, trans_time = transcribe_audio(tmpfile.name)
                        
                        if not transcription or len(transcription.strip()) < 2:
                            logger.warning("No speech detected, retrying...")
                            rep -= 1  # Retry this repetition
                            continue
                        
                        logger.info(f"Transcribed: '{transcription}'")
                        
                        # Process query
                        result = process_single_query(transcription, use_rag=use_rag)
                        result['question_id'] = f"q{q_idx}"
                        result['question_text'] = question
                        result['repetition'] = rep + 1
                        result['recording_time'] = rec_time
                        result['transcription_time'] = trans_time
                        result['transcription'] = transcription
                        
                        logger.info(f"Response: {result['response']}")
                        
                        # Speak response
                        if result['success']:
                            tts_time = text_to_speech(result['response'])
                            result['tts_time'] = tts_time
                        
                        all_results.append(result)
                        
                        # Small delay
                        time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Error in test: {e}")
                    continue
        
        # Break after every 10 questions
        if q_idx % 10 == 0 and q_idx < len(questions):
            logger.info("\n" + "="*60)
            logger.info("BREAK TIME - Rest for 2-3 minutes")
            logger.info("="*60)
            input("Press Enter when ready to continue...")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = Config.RESULTS_DIR / f'voice_test_results_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    perf_tracker.save_metrics()
    
    return all_results

# ============================================================================
# TEXT-BASED TESTING (for quick validation)
# ============================================================================

def run_text_test(questions: List[str], use_rag: bool = True) -> List[Dict]:
    """Text-based testing (no audio I/O) for quick validation"""
    logger.info("="*60)
    logger.info("TEXT TESTING MODE")
    logger.info("="*60)
    
    if not check_ollama_server():
        logger.error("Ollama server not running!")
        return []
    
    results = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"\n[{i}/{len(questions)}] {question}")
        result = process_single_query(question, use_rag=use_rag)
        results.append(result)
        logger.info(f"Response: {result.get('response', 'ERROR')}")
        time.sleep(0.5)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_str = "rag" if use_rag else "nonrag"
    results_file = Config.RESULTS_DIR / f'text_test_{mode_str}_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Quick test with sample questions
    sample_questions = [
        "What is a resistor?",
        "Explain PWM briefly",
        "Is LED a diode? True or false?"
    ]
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        logger.info("Running quick test...")
        run_text_test(sample_questions, use_rag=True)
    else:
        logger.info("Interactive mode not implemented in this version")
        logger.info("Use testing notebooks for guided testing")
