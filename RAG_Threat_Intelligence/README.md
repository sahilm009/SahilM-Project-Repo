RAG-Powered Threat Intelligence Assistant


Overview


This project implements a Retrieval-Augmented Generation (RAG) system for cybersecurity threat intelligence analysis. RAG combines vector-based document retrieval with large language model generation to provide accurate, contextual answers grounded in real data. The system queries a local knowledge base of threat intelligence, retrieves relevant documents through semantic search, and synthesizes natural language responses using a local LLM.
The assistant provides instant access to CVE vulnerabilities, MITRE ATT&CK techniques, malware family profiles, and threat actor intelligence through natural language queries. It automatically aggregates data from the National Vulnerability Database and MITRE ATT&CK framework into a searchable vector store. Users can ask questions about specific vulnerabilities, attack techniques, threat actors, or malware families and receive detailed technical summaries with source citations.
Requirements

Python 3.8+
Ollama (https://ollama.ai)

Installation


bashpip install -r requirements.txt
ollama pull llama3.2
python threat_intel_assistant.py


Security Operations Context


This project demonstrates capabilities directly applicable to enterprise security operations and threat intelligence platforms. The architecture mirrors commercial SIEM/SOAR solutions that enrich security alerts with threat intelligence context during incident investigation. Modern XDR and EDR platforms rely on similar retrieval systems to correlate endpoint telemetry with known threat actor TTPs and malware behaviors.
The implementation showcases skills in threat intelligence operations, including working with structured threat data from industry-standard sources (NVD, MITRE ATT&CK), building semantic search over security indicators, and automating the kind of research that security analysts perform manually during investigations. These are core competencies for security engineering roles focused on detection, response, and threat hunting.


Technical Architecture


Data Ingestion: Pulls CVE data from the National Vulnerability Database API and MITRE ATT&CK techniques from the official repository. Includes custom threat actor and malware family profiles.
Vector Storage: Converts threat intelligence documents into 300-dimensional embeddings using hash-based vectorization. Implements cosine similarity search for semantic retrieval.
RAG Pipeline: Processes queries through three stages - retrieve relevant documents via vector search, augment retrieved context into a structured prompt, and generate natural language responses using Ollama's local LLM.
Offline Operation: Caches all threat intelligence locally after initial download. Subsequent queries execute entirely offline with no external API dependencies.
Use Cases
Alert Enrichment: Automatically research CVEs, techniques, and IOCs that appear in SIEM alerts to provide context for security analysts.
Threat Hunting: Query relationships between malware families, threat actors, and attack techniques during proactive hunting exercises.
Incident Response: Rapidly access threat intelligence during active incidents to understand attacker TTPs and predict lateral movement.
Vulnerability Management: Research CVE details, exploitation methods, and threat actor interest to prioritize patching decisions.


Example Queries

"What is CVE-2024-3094 and how is it exploited?"
"Which threat actors use spearphishing attachments?"
"Describe Emotet malware and its TTPs"
"What MITRE techniques involve PowerShell execution?"
"How does LockBit ransomware operate?"


Knowledge Base Contents

CVE Database: Vulnerabilities with CVSS scores, descriptions, and publication dates
MITRE ATT&CK: Tactics, techniques, and procedures mapped to attack stages
Malware Profiles: Family classifications, TTPs, and indicators of compromise
Threat Actors: Attribution data, targeting patterns, and known campaigns
