#!/bin/bash
# CareerLens — One-command setup script
set -e

echo ""
echo "◈ CareerLens Setup"
echo "══════════════════════════════════════"

# 1. Create virtual environment
echo "→ Creating virtual environment..."
python -m venv venv
source venv/bin/activate || . venv/Scripts/activate

# 2. Upgrade pip
pip install --upgrade pip -q

# 3. Install requirements
echo "→ Installing dependencies (this may take 2–5 min on first run)..."
pip install -r backend/requirements.txt -q

# 4. Download spaCy model
echo "→ Downloading spaCy language model..."
python -m spacy download en_core_web_sm -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run CareerLens:"
echo "  source venv/bin/activate"
echo "  cd backend"
echo "  python main.py"
echo ""
echo "Then open: http://localhost:8000"