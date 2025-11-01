# SunnySett AI Model Discovery Platform

A comprehensive collection of pre-trained models across 10 categories for AI model discovery and customization. This platform provides ready-to-use Python scripts for deploying and testing various AI models on GCP VMs.

## 🚀 Quick Start

1. **Clone or Download** this repository
2. **Navigate** to any model directory: `cd sunnysett_models/Marketing/`
3. **Run** a model script: `python bert_base_uncased.py`
4. **Customize** the scripts for your specific use cases

## 📊 Platform Overview

| Category | Model Count | Description |
|----------|-------------|-------------|
| **Marketing** | 5 | Text classification, sentiment analysis, and content generation |
| **Finance** | 5 | Financial text analysis, sentiment analysis, and risk assessment |
| **Material Science** | 5 | Scientific text understanding and material property prediction |
| **Engineering** | 5 | Technical document analysis, code understanding, and engineering |
| **Cybersecurity** | 5 | Threat detection, malware analysis, and security text classification |
| **Healthcare** | 5 | Biomedical text analysis, clinical document understanding, and medical NER |
| **Supply Chain** | 5 | Supply chain document processing, forecasting, and logistics |
| **Education** | 5 | Educational content analysis, question answering, and learning |
| **Law** | 5 | Legal document analysis, contract understanding, and legal text classification |
| **Climate** | 5 | Climate research analysis, environmental text understanding, and sustainability |

**Total: 50+ Pre-trained Models**

## 🏗️ Project Structure

```
sunnysett_models/
├── Marketing/
│   ├── bert_base_uncased.py
│   ├── facebook_bart_large_mnli.py
│   ├── Salesforce_blip_image_captioning.py
│   ├── cardiffnlp_twitter_roberta_base_sentiment_latest.py
│   └── microsoft_DialoGPT_medium.py
├── Finance/
│   ├── ProsusAI_finbert.py
│   ├── yiyanghkust_finbert_tone.py
│   ├── mrm8488_distilroberta_finetuned_financial_news_sentiment.py
│   ├── nlpaueb_financial_bert.py
│   └── microsoft_DialoGPT_medium.py
├── Material_Science/
│   ├── materialsproject_mp_bert.py
│   ├── schwallie_chemberta.py
│   ├── CompVis_stable_diffusion_v1_4.py
│   ├── allenai_scibert_scivocab_uncased.py
│   └── microsoft_layoutlmv3_base.py
├── Engineering/
│   ├── sentence_transformers_all_MiniLM_L6_v2.py
│   ├── microsoft_layoutlmv3_base.py
│   ├── microsoft_codebert_base.py
│   ├── Salesforce_codet5_base.py
│   └── facebook_bart_large_cnn.py
├── Cybersecurity/
│   ├── cybersecurity_nlp_bert_malware.py
│   ├── roberta_cybersec.py
│   ├── symanto_sn_xlm_roberta_base_snli_mnli_anli_xnli.py
│   ├── microsoft_DialoGPT_medium.py
│   └── facebook_bart_large_mnli.py
├── Healthcare/
│   ├── emilyalsentzer_Bio_ClinicalBERT.py
│   ├── d4data_biomedical_ner_all.py
│   ├── cambridgeltl_SapBERT_from_PubMedBERT_fulltext.py
│   ├── microsoft_BiomedNLP_PubMedBERT_base_uncased_abstract_fulltext.py
│   └── facebook_bart_large_cnn.py
├── Supply_Chain/
│   ├── Salesforce_codet5_base.py
│   ├── t5_base.py
│   ├── sentence_transformers_all_MiniLM_L6_v2.py
│   ├── facebook_bart_large_mnli.py
│   └── microsoft_layoutlmv3_base.py
├── Education/
│   ├── deepset_roberta_base_squad2.py
│   ├── openai_whisper_base.py
│   ├── facebook_bart_large_cnn.py
│   ├── sentence_transformers_all_MiniLM_L6_v2.py
│   └── microsoft_DialoGPT_medium.py
├── Law/
│   ├── nlpaueb_legal_bert_base_uncased.py
│   ├── zlucia_legalbert.py
│   ├── bvanaken_bert_arch.py
│   ├── facebook_bart_large_mnli.py
│   └── microsoft_layoutlmv3_base.py
└── Climate/
    ├── climatebert_distilbert_base_uncased_finetuned_climate.py
    ├── allenai_scibert_scivocab_uncased.py
    ├── facebook_bart_large_cnn.py
    ├── sentence_transformers_all_MiniLM_L6_v2.py
    └── microsoft_layoutlmv3_base.py
```

## 🔧 Features

- **Ready-to-Run Scripts**: Each model has a complete Python script with dependencies
- **Automatic Installation**: Scripts automatically install required packages
- **GCP VM Compatible**: All scripts are optimized for cloud deployment
- **Comprehensive Examples**: Each script includes sample inference examples
- **Metadata Tracking**: CSV and JSON files track all models and their details
- **Category Organization**: Models organized by domain for easy discovery

## 📋 Prerequisites

- Python 3.10+
- pip package manager
- Internet connection (for downloading models)
- Hugging Face account (optional, for some models)

## 🚀 Usage Examples

### Running a Single Model

```bash
# Navigate to a specific model
cd sunnysett_models/Marketing/

# Run the BERT model
python bert_base_uncased.py
```

### Running All Models in a Category

```bash
# Run all marketing models
cd sunnysett_models/Marketing/
for script in *.py; do python "$script"; done
```

### Customizing Models

Each script is designed to be easily customizable:

1. **Modify Sample Data**: Change the sample texts in the `run_inference()` function
2. **Add New Features**: Extend the scripts with additional functionality
3. **Fine-tune Models**: Use the loaded models for fine-tuning on your data
4. **Deploy to Production**: Modify scripts for production deployment

## 📊 Model Categories

### Marketing
- **BERT Base Uncased**: General text classification
- **BART Large MNLI**: Zero-shot classification
- **BLIP Image Captioning**: Image-to-text generation
- **Twitter RoBERTa**: Social media sentiment analysis
- **DialoGPT**: Conversational AI

### Finance
- **FinBERT**: Financial text analysis
- **FinBERT Tone**: Financial sentiment analysis
- **DistilRoBERTa Financial**: Financial news sentiment
- **Financial BERT**: Financial text understanding
- **DialoGPT**: Financial chatbots

### Material Science
- **MP-BERT**: Materials science literature
- **ChemBERTa**: Chemical text understanding
- **Stable Diffusion**: Material visualization
- **SciBERT**: Scientific text understanding
- **LayoutLMv3**: Document understanding

### Engineering
- **Sentence Transformers**: Semantic search
- **LayoutLMv3**: Technical document understanding
- **CodeBERT**: Code understanding
- **CodeT5**: Code generation
- **BART Large CNN**: Text summarization

### Cybersecurity
- **BERT Malware**: Malware detection
- **RoBERTa Cybersec**: Security text analysis
- **XLM-RoBERTa**: Multilingual security analysis
- **DialoGPT**: Security chatbots
- **BART MNLI**: Zero-shot threat classification

### Healthcare
- **Bio ClinicalBERT**: Clinical text understanding
- **Biomedical NER**: Medical entity recognition
- **SapBERT**: Biomedical entity linking
- **PubMedBERT**: Biomedical text understanding
- **BART CNN**: Medical document summarization

### Supply Chain
- **CodeT5**: Supply chain automation
- **T5 Base**: Text-to-text tasks
- **Sentence Transformers**: Document similarity
- **BART MNLI**: Event classification
- **LayoutLMv3**: Document processing

### Education
- **RoBERTa SQuAD2**: Question answering
- **Whisper**: Speech recognition
- **BART CNN**: Content summarization
- **Sentence Transformers**: Content similarity
- **DialoGPT**: Educational chatbots

### Law
- **Legal BERT**: Legal text understanding
- **LegalBERT**: Legal document analysis
- **BERT-ARCH**: Legal argument mining
- **BART MNLI**: Legal text classification
- **LayoutLMv3**: Legal document understanding

### Climate
- **ClimateBERT**: Climate text analysis
- **SciBERT**: Scientific climate literature
- **BART CNN**: Climate research summarization
- **Sentence Transformers**: Document similarity
- **LayoutLMv3**: Climate report understanding

## 📁 Metadata Files

- **`sunnysett_models_metadata.csv`**: CSV format with all model information
- **`sunnysett_models_metadata.json`**: JSON format with structured metadata
- **`sunnysett_models_summary.md`**: Markdown summary with detailed tables

## 🔒 Security Notes

- API keys are stored in `.env` file (not committed to version control)
- All scripts include proper error handling
- Models are downloaded securely from Hugging Face
- No sensitive data is hardcoded in scripts

## 🚀 GCP Deployment

All scripts are ready for GCP VM deployment:

1. **Upload** the `sunnysett_models` directory to your GCP VM
2. **Install** Python 3.10+ on the VM
3. **Run** any model script: `python model_script.py`
4. **Scale** as needed for production workloads

## 🤝 Contributing

To add new models:

1. **Add** model information to `model_database.json`
2. **Run** `python generate_all_models.py` to create scripts
3. **Test** the new model script
4. **Update** metadata files

## 📄 License

This project is licensed under the MIT License. Individual models may have their own licenses - check the Hugging Face model pages for details.

## 🆘 Support

For issues or questions:

1. Check the model's Hugging Face page for documentation
2. Review the script comments for usage instructions
3. Ensure all dependencies are properly installed
4. Check your internet connection for model downloads

## 🎯 Next Steps

- **Fine-tune** models on your specific data
- **Deploy** to production environments
- **Integrate** with your existing applications
- **Scale** based on your requirements

---

**SunnySett AI Model Discovery Platform** - Making AI model deployment simple and accessible! 🚀
