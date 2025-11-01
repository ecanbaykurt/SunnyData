# SunnyData

Curated library of 141 ready-to-run model scripts spanning 33 categories and multiple modalities. Use these assets to explore, prototype, and deploy NLP, vision, audio, tabular, and multimodal workloads with minimal setup.

## 🚀 Quick Start

- Clone or download the repository
- Navigate to a model directory, for example `cd sunnysett_models/Marketing`
- Run the desired script: `python bert_base_uncased.py`
- Review the console output for dependency installation and sample inference

## 📦 Repository Layout

```
Pre trained model dataset/
├── sunnysett_models/                # Primary library (33 categories / 141 scripts)
├── sunnysett_gcp_ready/             # Legacy 10-category bundle kept for reference
├── sunnysett_models_metadata.csv    # Auto-generated metadata (CSV)
├── sunnysett_models_metadata.json   # Auto-generated metadata (JSON)
├── sunnysett_models_summary.md      # Inventory statistics and category breakdown
├── add_specialized_models.py        # Generator for persona and domain bundles
├── analyze_and_organize.py          # Audit/cleanup helper for model scripts
└── analysis_model_summary.json      # Derived analytics powering the summary files
```

### Category Highlights

- 33 categories covering industries, personas, deployment profiles, and core modalities
- 135 neural/deep learning models plus 6 classical or AutoML forecasters
- Coverage spans Text/NLP, Vision, Audio, Tabular/Time-Series, and Multimodal pipelines
- Full counts and modality breakdowns live in `sunnysett_models_summary.md`

## 📊 Inventory Snapshot

- **Total Categories**: 33
- **Total Models**: 141
- **Neural / Deep Learning**: 135
- **Classical / AutoML**: 6
- **Top Modalities**: Text/NLP (91), Vision (15), Audio (12), Tabular/Time-Series (12), Vision+Text (11)
- For a complete table with modality and type splits per category, open `sunnysett_models_summary.md`

## 🛠️ Running the Scripts

- Each script auto-installs required packages on first run (via `pip`)
- Sample inference calls live inside `run_inference` functions—edit the sample payloads to try your own data
- To execute an entire category on Linux/macOS:
  - `cd sunnysett_models/<Category>`
  - `for script in *.py; do python "$script"; done`
- On Windows PowerShell:
  - `Get-ChildItem *.py | ForEach-Object { python $_ }`

## 🗂️ Metadata & Tooling

- `sunnysett_models_metadata.json` / `.csv`: regenerated automatically from the codebase; reflects every script with category, pipeline, modality, and Hugging Face URL inference
- `analysis_model_summary.json`: consolidated analytics (counts, modality mixes, type splits) consumed by the documentation
- `add_specialized_models.py`: reproduces persona and deployment bundles if you need to regenerate scripts
- `analyze_and_organize.py`: scans the library, reports duplicates, and highlights any gaps relative to catalog CSVs

## 🔁 Adding or Updating Models

- Drop a new script in the relevant subdirectory under `sunnysett_models/<Category>`
- Follow the existing template: docstring metadata, `install_dependencies`, `load_model`, and `run_inference`
- Run `python analyze_and_organize.py` to verify classification and metadata coverage
- After changes, run `python refresh_metadata.py` to regenerate the summary, CSV, and JSON inventories

## ☁️ Deployment Notes

- Scripts are tested with Python 3.10+ and install dependencies on demand
- For GCP or other cloud VMs, copy either `sunnysett_models/` (expanded library) or `sunnysett_gcp_ready/` (compact legacy bundle)
- Ensure outbound internet access so Hugging Face models can download weights on first execution

## 🤝 Contributing

- Follow the scripting convention already in place (docstring, helper functions, clean logging)
- Update `sunnysett_models_summary.md` and metadata files by re-running the generation helper after adding assets
- Validate new models locally before proposing PRs or deploying to cloud environments

## 📄 License

This project is released under the MIT License. Individual model checkpoints may carry additional licensing requirements—consult each model’s Hugging Face page.

## 🆘 Getting Help

- Review the docstring in each script for usage instructions and dependency notes
- Verify network access and Hugging Face availability if downloads fail
- Open an issue or discussion thread with reproducible steps if you uncover defects

---

**SunnySett AI Model Discovery Platform** — rapid exploration of production-ready model starters.
