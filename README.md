# Accessing Privacy Policies using LLMs 

Framework for evaluating Large Language Models on the **C3PA dataset** of privacy policy annotations (CCPA/CPRA).

## Features
- Preprocessing and span filtering
- Context extraction from privacy policies
- Multi-model inference (OpenAI, Gemini, Anthropic, DeepSeek, Groq, Ollama, OpenRouter)
- Evaluation reports in console, JSON, or DOCX

## Download
```bash
git clone https://github.com/zhovka02/AccessingPrivacyPoliciesusingLLMs.git
cd AccessingPrivacyPoliciesusingLLMs
pip install -r requirements.txt

## Environment
API keys are loaded from a .env file. Example:
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
