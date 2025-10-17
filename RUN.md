# Quick Start Guide - 5 Minutes to First Poem

## Step 1: Get Groq API Key (2 minutes)

1. Visit: https://console.groq.com/keys
2. Sign up (free, no credit card)
3. Click "Create API Key"
4. Copy your key (starts with `gsk_...`)

## Step 2: Install (1 minute)

```bash
# Clone repository
git clone <your-repo-url>
cd collaborative-poetry-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Configure (30 seconds)

```bash
# Set your API key
export GROQ_API_KEY='gsk_your_key_here'

# Windows users:
# set GROQ_API_KEY=gsk_your_key_here
```

## Step 4: Run (1 minute)

```bash
python poetry_system.py
```

That's it! You'll see:
- Document processing
- Fact extraction
- 8 verses created alternately by two poets
- Detailed judgment with scores
- Winner announcement

## What You Get

```
================================================================================
                           COLLABORATIVE POEM
================================================================================

1. [Poet A (Metaphorical)]
   Like eagles breaking dawn's silence, two souls pierced the lunar veil.

2. [Poet B (Narrative)]
   Armstrong's boot marked July 20, 1969, in dust forever still.
   
...

================================================================================
                               JUDGMENTS
================================================================================

Poet A (Metaphorical)
----------------------------------------
Overall Score: 8.35/10

Detailed Scores:
  Factual Accuracy: 8.5/10
  Poetic Quality: 9.0/10
  ...

WINNER: Poet A (Metaphorical)
```

## Try Your Own Document

```bash
# Create a text file
echo "Your content here..." > my_document.txt

# Generate poem
python -c "
from poetry_system import generate_collaborative_poem, display_results
results = generate_collaborative_poem('my_document.txt', num_verses=6)
display_results(results)
"
```

## Supported File Types

- Text files (`.txt`)
- PDFs (`.pdf`)
- Word documents (`.docx`)
- Images (`.jpg`, `.png` - requires Tesseract)

## Next Steps

- **More examples**: `python example_usage.py`
- **Audio output**: Install `pip install gTTS` then use audio_generator.py
- **Full docs**: Read README.md
- **Design details**: Read DESIGN.md

## Common Issues

**"GROQ_API_KEY not set"**
```bash
export GROQ_API_KEY='your-key-here'
```

**"Module not found: langchain_groq"**
```bash
pip install -r requirements.txt
```

**"Tesseract not found" (for images)**
```bash
# macOS: brew install tesseract
# Ubuntu: sudo apt-get install tesseract-ocr
# Windows: Download from GitHub
```

## Quick Commands

```bash
# Basic usage
python poetry_system.py

# With your own file
python poetry_system.py my_document.pdf

# Run examples
python example_usage.py

# Generate audio
python audio_generator.py
```

## Performance

- **Time**: 40 seconds for 8 verses
- **Cost**: FREE (Groq API)
- **Quality**: High (openai/gpt-oss-120b)

## Need Help?

1. Check **README.md** for detailed setup
2. Check **DESIGN.md** for how it works
3. Check **example_usage.py** for code examples

---

**That's it! You're ready to generate factually-grounded collaborative poetry.**