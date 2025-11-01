# SunnySett AI Model Discovery Platform

A comprehensive collection of pre-trained models across 10 categories for AI model discovery and customization.

## Summary Statistics

- **Total Categories**: 10
- **Total Models**: 50
- **Average Models per Category**: 5

## Category Overview

| Category | Model Count | Description |
|----------|-------------|-------------|
| Marketing | 5 | Text classification, sentiment analysis, and content generation for marketing applications |
| Finance | 5 | Financial text analysis, sentiment analysis, and risk assessment models |
| Material Science | 5 | Scientific text understanding and material property prediction models |
| Engineering | 5 | Technical document analysis, code understanding, and engineering applications |
| Cybersecurity | 5 | Threat detection, malware analysis, and security text classification |
| Healthcare | 5 | Biomedical text analysis, clinical document understanding, and medical NER |
| Supply Chain | 5 | Supply chain document processing, forecasting, and logistics optimization |
| Education | 5 | Educational content analysis, question answering, and learning applications |
| Law | 5 | Legal document analysis, contract understanding, and legal text classification |
| Climate | 5 | Climate research analysis, environmental text understanding, and sustainability |

## Detailed Model List

### Marketing

| Model Name | Task | Description |
|------------|------|-------------|
| `bert-base-uncased` | text-classification | General purpose BERT model for text classification and sentiment analysis |
| `facebook/bart-large-mnli` | zero-shot-classification | BART model fine-tuned on Multi-Genre Natural Language Inference for zero-shot classification |
| `Salesforce/blip-image-captioning` | image-to-text | BLIP model for image captioning and visual question answering |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | sentiment-analysis | RoBERTa model fine-tuned for Twitter sentiment analysis |
| `microsoft/DialoGPT-medium` | text-generation | DialoGPT model for conversational AI and chatbot applications |

### Finance

| Model Name | Task | Description |
|------------|------|-------------|
| `ProsusAI/finbert` | text-classification | BERT model pre-trained on financial communication text |
| `yiyanghkust/finbert-tone` | sentiment-analysis | FinBERT model fine-tuned for financial sentiment analysis |
| `mrm8488/distilroberta-finetuned-financial-news-sentiment` | sentiment-analysis | DistilRoBERTa model fine-tuned for financial news sentiment |
| `nlpaueb/financial-bert` | text-classification | Financial BERT model for financial text understanding |
| `microsoft/DialoGPT-medium` | text-generation | DialoGPT for financial chatbot applications |

### Material Science

| Model Name | Task | Description |
|------------|------|-------------|
| `materialsproject/mp-bert` | text-classification | BERT model trained on materials science literature |
| `schwallie/chemberta` | text-classification | ChemBERTa model for chemical text understanding |
| `CompVis/stable-diffusion-v1-4` | text-to-image | Stable Diffusion for generating material visualizations |
| `allenai/scibert_scivocab_uncased` | text-classification | SciBERT model for scientific text understanding |
| `microsoft/layoutlmv3-base` | document-question-answering | LayoutLMv3 for document understanding and material data extraction |

### Engineering

| Model Name | Task | Description |
|------------|------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | feature-extraction | Sentence transformer for semantic search and similarity |
| `microsoft/layoutlmv3-base` | document-question-answering | LayoutLMv3 for technical document understanding |
| `microsoft/codebert-base` | text-classification | CodeBERT for code understanding and generation |
| `Salesforce/codet5-base` | text2text-generation | CodeT5 for code generation and summarization |
| `facebook/bart-large-cnn` | summarization | BART model for text summarization of technical documents |

### Cybersecurity

| Model Name | Task | Description |
|------------|------|-------------|
| `cybersecurity-nlp/bert-malware` | text-classification | BERT model trained for malware detection and analysis |
| `roberta-cybersec` | text-classification | RoBERTa model fine-tuned for cybersecurity text analysis |
| `symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli` | text-classification | Multilingual RoBERTa for natural language inference in security contexts |
| `microsoft/DialoGPT-medium` | text-generation | DialoGPT for security chatbot and incident response |
| `facebook/bart-large-mnli` | zero-shot-classification | BART for zero-shot classification of security threats |

### Healthcare

| Model Name | Task | Description |
|------------|------|-------------|
| `emilyalsentzer/Bio_ClinicalBERT` | text-classification | Clinical BERT model for biomedical text understanding |
| `d4data/biomedical-ner-all` | token-classification | Biomedical NER model for extracting medical entities |
| `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | feature-extraction | SapBERT for biomedical entity linking and normalization |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | text-classification | PubMedBERT for biomedical text understanding |
| `facebook/bart-large-cnn` | summarization | BART for medical document summarization |

### Supply Chain

| Model Name | Task | Description |
|------------|------|-------------|
| `Salesforce/codet5-base` | text2text-generation | CodeT5 for supply chain code analysis and automation |
| `t5-base` | text2text-generation | T5 model for text-to-text tasks in supply chain management |
| `sentence-transformers/all-MiniLM-L6-v2` | feature-extraction | Sentence transformer for supply chain document similarity |
| `facebook/bart-large-mnli` | zero-shot-classification | BART for zero-shot classification of supply chain events |
| `microsoft/layoutlmv3-base` | document-question-answering | LayoutLMv3 for processing supply chain documents and forms |

### Education

| Model Name | Task | Description |
|------------|------|-------------|
| `deepset/roberta-base-squad2` | question-answering | RoBERTa model fine-tuned for question answering |
| `openai/whisper-base` | automatic-speech-recognition | Whisper model for speech recognition and transcription |
| `facebook/bart-large-cnn` | summarization | BART for educational content summarization |
| `sentence-transformers/all-MiniLM-L6-v2` | feature-extraction | Sentence transformer for educational content similarity |
| `microsoft/DialoGPT-medium` | text-generation | DialoGPT for educational chatbots and tutoring |

### Law

| Model Name | Task | Description |
|------------|------|-------------|
| `nlpaueb/legal-bert-base-uncased` | text-classification | Legal BERT model for legal text understanding |
| `zlucia/legalbert` | text-classification | LegalBERT model for legal document analysis |
| `bvanaken/bert-arch` | text-classification | BERT-ARCH for legal argument mining |
| `facebook/bart-large-mnli` | zero-shot-classification | BART for zero-shot legal text classification |
| `microsoft/layoutlmv3-base` | document-question-answering | LayoutLMv3 for legal document understanding |

### Climate

| Model Name | Task | Description |
|------------|------|-------------|
| `climatebert/distilbert-base-uncased-finetuned-climate` | text-classification | ClimateBERT model for climate-related text analysis |
| `allenai/scibert_scivocab_uncased` | text-classification | SciBERT for scientific climate literature analysis |
| `facebook/bart-large-cnn` | summarization | BART for climate research summarization |
| `sentence-transformers/all-MiniLM-L6-v2` | feature-extraction | Sentence transformer for climate document similarity |
| `microsoft/layoutlmv3-base` | document-question-answering | LayoutLMv3 for climate report document understanding |

## Usage Instructions

1. **Install Dependencies**: Each script automatically installs required packages
2. **Run Models**: Execute individual Python scripts for specific models
3. **Customize**: Modify scripts for your specific use cases
4. **Deploy**: All scripts are ready for GCP VM deployment

## File Structure

```
sunnysett_models/
├── Marketing/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Finance/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Material_Science/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Engineering/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Cybersecurity/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Healthcare/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Supply_Chain/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Education/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Law/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── Climate/
│   ├── model1.py
│   ├── model2.py
│   └── ...
├── sunnysett_models_metadata.csv
├── sunnysett_models_metadata.json
└── sunnysett_models_summary.md
```
