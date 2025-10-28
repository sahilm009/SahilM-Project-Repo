"""
RAG-Powered Threat Intelligence Assistant
A simple local chatbot for querying CVE, MITRE ATT&CK, and threat intelligence data
Uses Ollama for local LLM generation
"""

import json
import requests
import numpy as np
from typing import List, Dict, Optional
import pickle
import os

# Simple vector store using numpy
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
    
    def add_documents(self, texts: List[str], metadata: List[Dict]):
        """Add documents with simple TF-IDF style embeddings"""
        for text, meta in zip(texts, metadata):
            embedding = self._simple_embed(text)
            self.documents.append(text)
            self.embeddings.append(embedding)
            self.metadata.append(meta)
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """Create a simple bag-of-words embedding"""
        words = text.lower().split()
        embedding = np.zeros(300)
        for word in words:
            idx = hash(word) % 300
            embedding[idx] += 1
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if not self.embeddings:
            return []
        
        query_emb = self._simple_embed(query)
        embeddings_array = np.array(self.embeddings)
        
        # Cosine similarity
        similarities = np.dot(embeddings_array, query_emb)
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_idx:
            results.append({
                'text': self.documents[idx],
                'metadata': self.metadata[idx],
                'score': float(similarities[idx])
            })
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'metadata': self.metadata
            }, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']


class ThreatIntelligenceRAG:
    def __init__(self, cache_dir: str = "./threat_intel_cache"):
        self.vector_store = SimpleVectorStore()
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.vector_store_path = os.path.join(cache_dir, "vector_store.pkl")
        self.ollama_model = "llama3.2:3b" 
    
    def load_cve_data(self, limit: int = 100):
        """Load recent CVE data from NVD API"""
        print("Getting data from NVD...")
        
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        params = {"resultsPerPage": limit}
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            documents = []
            metadata = []
            
            for item in data.get('vulnerabilities', [])[:limit]:
                cve = item.get('cve', {})
                cve_id = cve.get('id', 'Unknown')
                
                descriptions = cve.get('descriptions', [])
                description = descriptions[0].get('value', 'No description') if descriptions else 'No description'
                
                metrics = cve.get('metrics', {})
                cvss_score = 'N/A'
                if 'cvssMetricV31' in metrics and metrics['cvssMetricV31']:
                    cvss_score = metrics['cvssMetricV31'][0].get('cvssData', {}).get('baseScore', 'N/A')
                
                published = cve.get('published', 'Unknown')
                
                doc_text = f"CVE ID: {cve_id}\nDescription: {description}\nCVSS Score: {cvss_score}\nPublished: {published}"
                
                documents.append(doc_text)
                metadata.append({
                    'type': 'CVE',
                    'id': cve_id,
                    'score': cvss_score,
                    'published': published
                })
            
            self.vector_store.add_documents(documents, metadata)
            print(f"Loaded {len(documents)} CVE records")
            
        except Exception as e:
            print(f"Error fetching CVE data: {e}")
    
    def load_mitre_attack_data(self, limit: int = 50):
        """Load MITRE ATT&CK techniques"""
        print("Getting MITRE ATT&CK data...")
        
        url = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            documents = []
            metadata = []
            
            count = 0
            for obj in data.get('objects', []):
                if count >= limit:
                    break
                
                if obj.get('type') == 'attack-pattern':
                    technique_id = obj.get('external_references', [{}])[0].get('external_id', 'Unknown')
                    name = obj.get('name', 'Unknown')
                    description = obj.get('description', 'No description')
                    
                    tactics = [phase.get('phase_name', '') for phase in obj.get('kill_chain_phases', [])]
                    
                    doc_text = f"Technique: {technique_id} - {name}\nTactics: {', '.join(tactics)}\nDescription: {description[:500]}"
                    
                    documents.append(doc_text)
                    metadata.append({
                        'type': 'MITRE_ATTACK',
                        'technique_id': technique_id,
                        'name': name,
                        'tactics': tactics
                    })
                    count += 1
            
            self.vector_store.add_documents(documents, metadata)
            print(f"Loaded {len(documents)} MITRE ATT&CK techniques")
            
        except Exception as e:
            print(f"Error fetching MITRE ATT&CK data: {e}")
    
    def add_custom_threat_data(self):
        """Add custom threat intelligence examples, just in case of failure in getting data"""
        print("Adding custom threat intelligence...")
        
        custom_data = [
            {
                'text': "Malware Family: Emotet\nType: Banking Trojan\nTTPs: Uses malicious macros in Office documents, T1566.001 (Phishing: Spearphishing Attachment), T1204.002 (User Execution: Malicious File), T1059.003 (Command and Scripting Interpreter: Windows Command Shell)\nIOCs: Uses PowerShell to download additional payloads, C2 communication over HTTP/HTTPS",
                'metadata': {'type': 'MALWARE', 'family': 'Emotet', 'category': 'Banking Trojan'}
            },
            {
                'text': "Malware Family: Cobalt Strike\nType: Post-Exploitation Framework\nTTPs: T1055 (Process Injection), T1071.001 (Application Layer Protocol: Web Protocols), T1027 (Obfuscated Files or Information)\nIOCs: Beacon implants, named pipes, reflective DLL injection",
                'metadata': {'type': 'MALWARE', 'family': 'Cobalt Strike', 'category': 'Post-Exploitation'}
            },
            {
                'text': "Threat Actor: APT29 (Cozy Bear)\nOrigin: Russia\nTargets: Government, diplomatic, think tank, healthcare entities\nTTPs: T1566.002 (Phishing: Spearphishing Link), T1059.001 (PowerShell), T1078 (Valid Accounts)\nKnown Campaigns: SolarWinds supply chain attack",
                'metadata': {'type': 'THREAT_ACTOR', 'name': 'APT29', 'origin': 'Russia'}
            },
            {
                'text': "Malware Family: LockBit\nType: Ransomware-as-a-Service\nTTPs: T1486 (Data Encrypted for Impact), T1490 (Inhibit System Recovery), T1027 (Obfuscated Files)\nIOCs: Double extortion tactics, .lockbit file extension, ransom notes named 'Restore-My-Files.txt'",
                'metadata': {'type': 'MALWARE', 'family': 'LockBit', 'category': 'Ransomware'}
            },
        ]
        
        documents = [item['text'] for item in custom_data]
        metadata = [item['metadata'] for item in custom_data]
        
        self.vector_store.add_documents(documents, metadata)
        print(f"Added {len(documents)} custom threat intelligence entries")
    
    def build_index(self):
        """Build the complete knowledge base"""
        print("\nBuilding Threat Intelligence Knowledge Base...\n")
        
        self.load_cve_data(limit=50)
        self.load_mitre_attack_data(limit=30)
        self.add_custom_threat_data()
        
        self.vector_store.save(self.vector_store_path)
        print(f"\nKnowledge base built and saved to {self.vector_store_path}")
    
    def load_index(self):
        """Load existing knowledge base from disk"""
        if os.path.exists(self.vector_store_path):
            print("Loading existing knowledge base...")
            self.vector_store.load(self.vector_store_path)
            print(f"Loaded {len(self.vector_store.documents)} documents")
            return True
        return False
    
    def check_ollama(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check if our model is available
            available = any(self.ollama_model in name for name in model_names)
            return available
        except:
            return False
    
    def generate_with_ollama(self, prompt: str) -> Optional[str]:
        """Generate response using Ollama"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get('response', '')
        except requests.exceptions.ConnectionError:
            return None
        except Exception as e:
            print(f"Ollama error: {e}")
            return None
    
    def query(self, question: str, k: int = 5) -> str:
        """Query the threat intelligence knowledge base using RAG"""
        
        #STEP 1: RETRIEVAL 
        print("  [1/3] Retrieving relevant documents...")
        results = self.vector_store.search(question, k=k)
        
        if not results:
            return "No relevant threat intelligence found for your query."
        
        #STEP 2: AUGMENTATION 
        print("  [2/3] Augmenting context...")
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"Document {i}:\n{result['text']}")
        
        context = "\n\n".join(context_parts)
        
        #STEP 3: GENERATION 
        print("  [3/3] Generating answer...")
        # Build prompt for LLM
        prompt = f"""You are a cybersecurity threat intelligence analyst. Answer the user's question using ONLY the provided threat intelligence context. Be concise and technical.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Include specific CVE IDs, technique IDs, or malware names when relevant
- If the context doesn't contain enough information, say so
- Keep the answer under 200 words
- Be technical and precise

ANSWER:"""
        
        # Generate with Ollama
        llm_response = self.generate_with_ollama(prompt)
        
        if llm_response:
            return f"""{llm_response}

---
Sources: {len(results)} documents retrieved from threat intelligence database"""
        else:
            # Fallback if Ollama not available
            return f"""Ollama not available. Showing retrieved documents:

{context}

---
 To get AI-generated summaries:
   1. Install Ollama: https://ollama.ai
   2. Run: ollama pull {self.ollama_model}
   3. Restart this assistant"""
    
    def interactive_mode(self):
        """Run interactive Q&A mode"""
        print("\n" + "="*60)
        print("  Threat Intelligence Assistant (RAG-Powered)")
        print("="*60)
        
        # Check Ollama status
        if self.check_ollama():
            print(f"Ollama connected | Model: {self.ollama_model}")
        else:
            print(f"Ollama not detected (install from https://ollama.ai)")
            print(f"Run: ollama pull {self.ollama_model}")
        
        print("\nCommands:")
        print("  - Type your question to search the knowledge base")
        print("  - Type 'rebuild' to refresh the data")
        print("  - Type 'quit' or 'exit' to stop")
        print("\n" + "="*60 + "\n")
        
        while True:
            try:
                query = input("\n Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n Goodbye!")
                    break
                
                if query.lower() == 'rebuild':
                    self.build_index()
                    continue
                
                print("\n Processing query...")
                response = self.query(query)
                print("\n" + response)
                
            except KeyboardInterrupt:
                print("\n\nExiting")
                break
            except Exception as e:
                print(f"\nError: {e}")


def main():
    """Main entry point"""
    rag = ThreatIntelligenceRAG()
    
    # Try to load existing index, otherwise build new one
    if not rag.load_index():
        print("No existing knowledge base found. Building new one...")
        rag.build_index()
    
    # Start interactive mode
    rag.interactive_mode()


if __name__ == "__main__":
    main()