# Collaborative AI Poetry System
### Using LangChain, LangGraph & Groq API

A sophisticated system where two AI agents with distinct poetic styles collaboratively create factually-grounded poetry from documents, with a third AI agent judging their contributions.

---

## System Overview

### What It Does
1. **Takes a document** (image, PDF, DOCX, or text file)
2. **Extracts factual information** (entities, facts, themes)
3. **Two AI poets collaborate** to create verses, alternating turns
   - Poet A: Metaphorical style (imagery, symbolism)
   - Poet B: Narrative style (story-driven, concrete)
4. **Judging agent evaluates** both poets on 5 criteria
5. **Declares winner** with detailed feedback

### Key Features
- **Functional** - Pure functional programming approach
- **LangChain** - For LLM orchestration and prompt templates
- **LangGraph** - For state management and workflow
- **Groq API** - Fast inference with Llama models
- **Factual Grounding** - All verses reference source material
- **Multi-criteria Judging** - Transparent evaluation framework
- **Bonus Audio** - Text-to-speech output

---

## Quick Start

### Prerequisites
- Python 3.9+
- Groq API key (get free at [console.groq.com](https://console.groq.com/keys))

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd collaborative-poetry-system

# 2. Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your Groq API key
export GROQ_API_KEY='your-groq-api-key-here'

# Windows users:
# set GROQ_API_KEY=your-groq-api-key-here

# 5. Run the system
python poetry_system.py
```

### Expected Output

```
Created sample document: sample_document.txt

Starting collaborative poetry generation...
================================================================================
Processing document: sample_document.txt
Extracting factual context...
Verse 1: Poet A (Metaphorical)
Verse 2: Poet B (Narrative)
Verse 3: Poet A (Metaphorical)
...
Judging contributions...

================================================================================
                           COLLABORATIVE POEM
================================================================================

1. [Poet A (Metaphorical)]
   Like eagles soaring through cosmic night, two souls touched the silver sphere.

2. [Poet B (Narrative)]
   Armstrong's boot marked July 20, 1969, in lunar dust forever.

...

WINNER: Poet A (Metaphorical)

Results saved to poem_results.json
```

---

## Project Structure

```
collaborative-poetry-system/
├── poetry_system.py           # Main system
├── poetry_system.ipynb
├── audio_generator.py         # Audio output
├── audio_generator.ipynb 
├── example_usage.py           # Example Usages
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── DESIGN.md                  # Design decisions & approach
├── Run_Locally.md             # Run locally
├── sample_document.txt        # Auto-generated sample
├── poem_results.json          # Output file
├── poem_audio.wav             # Generated Audio

```

---

## Usage Examples

### Basic Usage

```python
from poetry_system import generate_collaborative_poem, display_results

# Generate poem from any document
results = generate_collaborative_poem("my_document.pdf", num_verses=8)

# Display results
display_results(results)
```

### Custom Documents

```python
# Works with various formats
generate_collaborative_poem("research_paper.pdf", num_verses=10)
generate_collaborative_poem("article.txt", num_verses=6)
generate_collaborative_poem("scanned_page.jpg", num_verses=8)
```

### Accessing Results

```python
results = generate_collaborative_poem("document.txt")

# Access specific parts
factual_context = results['factual_context']
verses = results['verses']
judgments = results['judgments']
winner = results['winner']

# Each verse has:
# - line: The actual verse text
# - author: Which poet wrote it
# - line_number: Position in poem
# - factual_anchors: Facts referenced
```

---

## Bonus: Audio Output

Generate audio versions of poems with different voices:

```bash
# Install audio dependencies (if not already)
pip install gTTS pyttsx3

# Run audio generator
python audio_generator.py
```

Or programmatically:

```python
from audio_generator import AudioGenerator

# Generate audio
generator = AudioGenerator(engine_type="gtts")  # or "pyttsx3"
audio_path = generator.generate(verses, "poem.mp3")

# Play it
from audio_generator import play_audio
play_audio(audio_path)
```

---

## System Architecture

### Functional Design

The system uses pure functions organized in a data flow:

```
Document → Text Extraction → Fact Extraction → Verse Generation → Judging → Results
```

### LangGraph Workflow

```python
StateGraph:
  ├─ load_document_node
  ├─ extract_facts_node
  ├─ generate_verse_node (loops until max_verses)
  │   ├─ poet_a turn
  │   └─ poet_b turn
  └─ judge_poems_node → END
```

### LangChain Components

1. **Fact Extraction Chain**
   - Prompt template for structured fact extraction
   - JSON output parser
   - Returns: entities, facts, themes, temporal data, numbers

2. **Poet Chains** (x2)
   - Distinct personas for each poet
   - Context-aware prompts
   - String output parser

3. **Judging Chain**
   - Multi-criteria evaluation prompt
   - Structured feedback format
   - Comparative analysis

---

## Configuration

### Environment Variables

```bash
# Required
export GROQ_API_KEY='your-key-here'

# Optional (for debugging)
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY='your-langsmith-key'
```

### Model Selection

Edit `poetry_system.py` to change models:

```python
# Current default: openai/gpt-oss-120b
llm = ChatGroq(
    model="openai/gpt-oss-120b",  # Fast and high quality
    # model="llama-3.1-8b-instant",   # Faster, lower quality
    # model="mixtral-8x7b-32768",     # Alternative
    temperature=0.7,
    groq_api_key=os.getenv('GROQ_API_KEY')
)
```

### Adjusting Verse Count

```python
# Generate shorter or longer poems
results = generate_collaborative_poem("doc.txt", num_verses=4)   # Short
results = generate_collaborative_poem("doc.txt", num_verses=16)  # Long
```

---

## How It Works

### Phase 1: Document Processing

```python
def process_document(file_path: str) -> str
```

- Detects file type from extension
- Routes to appropriate processor:
  - Images → OCR with pytesseract
  - PDFs → PyPDF2 text extraction
  - DOCX → python-docx
  - TXT → direct read
- Returns plain text

### Phase 2: Fact Extraction

```python
def extract_facts_node(state: PoemState) -> Dict
```

- Uses LangChain with structured output
- Prompts Groq LLM to extract:
  - **Entities**: People, places, concepts
  - **Key Facts**: Concrete statements
  - **Themes**: Abstract topics
  - **Temporal**: Dates, sequences
  - **Numerical**: Stats, measurements
- Returns JSON structured data

### Phase 3: Collaborative Poetry Generation

```python
def generate_verse_node(state: PoemState) -> Dict
```

**LangGraph manages state flow:**
1. Poet A generates verse 1 (metaphorical style)
2. Poet B generates verse 2 (narrative style)
3. Poet A generates verse 3 (sees previous context)
4. Continue alternating...
5. Loop until `max_verses` reached

**Each poet:**
- Sees full factual context
- Reads all previous verses
- Maintains distinct style
- References specific facts
- Returns verse + facts used

### Phase 4: Judging

```python
def judge_poems_node(state: PoemState) -> Dict
```

**Evaluates both poets on 5 criteria (0-10 scale):**

1. **Factual Accuracy (25%)**: Grounding in source
2. **Poetic Quality (25%)**: Aesthetic merit
3. **Coherence (20%)**: Flow with poem
4. **Creativity (15%)**: Originality
5. **Emotional Impact (15%)**: Evocative power

**Returns for each poet:**
- Scores per criterion
- Overall weighted score
- Strengths (3+ points)
- Weaknesses (2+ points)
- Standout verses
- Detailed reasoning

**Winner**: Poet with higher overall score

---

## Design Philosophy

### Why Functional Programming?

**Traditional OOP Approach:**
```python
class PoetAgent:
    def __init__(self):
        self.style = ...
    def create_verse(self):
        ...
```

**Our Functional Approach:**
```python
def create_poet_chain(llm, poet_name, style):
    # Returns configured chain
    return prompt | llm | parser
```

**Benefits:**
- Simpler state management (LangGraph handles it)
- Easier to test (pure functions)
- More composable (chains as data)
- Better suited for LangChain/LangGraph patterns
- No hidden state or side effects

### Why LangChain?

**Without LangChain:**
```python
# Manual prompt construction
prompt = f"You are a poet... {context}... {previous_verses}..."
response = groq_api.chat(prompt)
# Manual parsing
```

**With LangChain:**
```python
# Reusable templates
prompt = ChatPromptTemplate.from_messages([...])
chain = prompt | llm | parser
result = chain.invoke({"context": context, ...})
```

**Benefits:**
- Reusable prompt templates
- Automatic output parsing
- Built-in error handling
- Easy to modify and extend
- Integration with LangGraph

### Why LangGraph?

**Without LangGraph:**
```python
# Manual state management
state = {"verses": [], "verse_num": 1}
while state["verse_num"] <= max_verses:
    verse = generate_verse(state)
    state["verses"].append(verse)
    state["verse_num"] += 1
```

**With LangGraph:**
```python
# Declarative workflow
workflow.add_node("generate_verse", generate_verse_node)
workflow.add_conditional_edges("generate_verse", check_completion, {...})
```

**Benefits:**
- Visual workflow representation
- Automatic state propagation
- Conditional branching
- Easy to add/modify nodes
- Built-in checkpointing (future)

### Why Groq API?

**Compared to OpenAI/Anthropic:**
- **10x faster** inference (LPU architecture)
- **Free tier** available
- **Great quality** with openai/gpt-oss-120b
- **Open source** models
- **Low latency** for real-time apps

**Perfect for:**
- Interactive applications
- Multiple API calls (our use case)
- Development and testing
- Cost-sensitive projects

---

## Judging Framework Design

### Multi-Criteria Evaluation

**Why 5 Criteria?**

Single score (e.g., "8/10") tells you nothing. Our framework provides:
- **Transparency**: See exactly what was evaluated
- **Actionability**: Know what to improve
- **Fairness**: Multiple perspectives prevent bias
- **Richness**: Captures poetry's complexity

### Criterion Breakdown

#### 1. Factual Accuracy (25%)
```
9-10: Multiple accurate facts, precise
7-8:  Clear factual grounding
5-6:  Vague references
3-4:  Weak connection
1-2:  Incorrect or no facts
```

**Why 25%?** Core requirement - poems must be grounded.

#### 2. Poetic Quality (25%)
```
9-10: Exceptional craft, memorable
7-8:  Strong imagery, good flow
5-6:  Competent but plain
3-4:  Clumsy or clichéd
1-2:  Poor language
```

**Why 25%?** Equal to factual accuracy - still must be poetry.

#### 3. Coherence (20%)
```
9-10: Seamless integration
7-8:  Good connections
5-6:  Somewhat connected
3-4:  Disjointed
1-2:  No connection
```

**Why 20%?** Slightly less critical but important for collaboration.

#### 4. Creativity (15%)
```
9-10: Genuinely surprising
7-8:  Notable originality
5-6:  Some fresh elements
3-4:  Predictable
1-2:  Clichéd
```

**Why 15%?** Enhancement, not requirement.

#### 5. Emotional Impact (15%)
```
9-10: Deeply moving
7-8:  Strong emotional pull
5-6:  Some feeling evoked
3-4:  Flat
1-2:  No connection
```

**Why 15%?** Enhancement, equal to creativity.

### Overall Score Formula

```python
overall = (
    0.25 × factual_accuracy +
    0.25 × poetic_quality +
    0.20 × coherence +
    0.15 × creativity +
    0.15 × emotional_impact
)
```

**Total weights = 100%**

---

## Troubleshooting

### Issue: "GROQ_API_KEY not found"

```bash
# Solution: Set environment variable
export GROQ_API_KEY='your-key-here'

# Or add to .env file
echo "GROQ_API_KEY=your-key-here" > .env
```

### Issue: "Module not found: langchain_groq"

```bash
# Solution: Install all dependencies
pip install -r requirements.txt

# Or specifically:
pip install langchain-groq
```

### Issue: "pytesseract not found" (for images)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue: Verses are too similar in style

**Solution**: Strengthen personas in code:

```python
# Edit create_poet_chain() function
personas = {
    "metaphorical": """MORE EXPLICIT INSTRUCTIONS HERE
    Use vivid metaphors, nature imagery, symbolism...
    NEVER be literal or straightforward...""",
    
    "narrative": """MORE EXPLICIT INSTRUCTIONS HERE
    Tell a clear story, use concrete details...
    NEVER be abstract or metaphorical..."""
}
```

### Issue: Low factual accuracy scores

**Solution**: Better fact extraction or explicit instructions:

```python
# In generate_verse_node(), add:
f"REQUIRED: Your verse MUST reference at least 2 specific facts: {facts}"
```

### Issue: Groq rate limits

```bash
# Free tier limits:
# - 30 requests/minute
# - 14,400 requests/day

# Solution: Add delays between requests
import time
time.sleep(2)  # 2 second delay
```

---

## Performance

### Execution Times (8-verse poem)

| Phase | Time | API Calls |
|-------|------|-----------|
| Document processing | 0.5-5s | 0 |
| Fact extraction | 2-5s | 1 |
| Verse generation | 2-4s per verse (16-32s) | 8 |
| Judging | 3-6s per poet (6-12s) | 2 |
| **Total** | **25-55s** | **11** |

### Cost Estimation (Groq Free Tier)

- Free tier: 14,400 requests/day
- Per poem: 11 requests
- **Can generate ~1,300 poems/day for free**

### Optimization Tips

1. **Use faster model for extraction:**
```python
# In extract_facts_node()
llm = ChatGroq(model="openai/gpt-oss-120b")  # 3x faster
```

2. **Reduce verse count:**
```python
generate_collaborative_poem("doc.txt", num_verses=4)  # 50% faster
```

3. **Cache fact extraction:**
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_extract_facts(text_hash):
    # Only extract once per unique document
    return extract_facts(text)
```

---

## Advanced Usage

### Batch Processing

```python
import glob

# Process all PDFs in a folder
for pdf_path in glob.glob("documents/*.pdf"):
    results = generate_collaborative_poem(pdf_path, num_verses=6)
    
    # Save with filename
    output = f"output/{Path(pdf_path).stem}_poem.json"
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
```

### Custom Poet Styles

Edit `create_poet_chain()` to add new styles:

```python
personas = {
    "metaphorical": "...",
    "narrative": "...",
    "haiku": """You write in haiku style:
    - 3 lines: 5-7-5 syllables
    - Nature imagery
    - Present tense
    - Seasonal reference""",
}
```

### Add More Poets

Modify the workflow to include a third poet:

```python
# In generate_verse_node()
if verse_num % 3 == 0:
    poet_name = "Poet C (Haiku)"
    style = "haiku"
elif verse_num % 3 == 1:
    poet_name = "Poet A (Metaphorical)"
    style = "metaphorical"
else:
    poet_name = "Poet B (Narrative)"
    style = "narrative"
```

### Export to Different Formats

```python
def export_poem_markdown(results, output_path):
    """Export poem as Markdown"""
    lines = ["# Collaborative Poem\n"]
    
    for verse in results['verses']:
        lines.append(f"**{verse['author']}**  ")
        lines.append(f"*{verse['line']}*\n")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

# Usage
export_poem_markdown(results, "poem.md")
```

---

## Key Design Decisions

### 1. Sequential vs Parallel Generation

**Chosen**: Sequential (poets alternate)

**Why**: 
- Natural coherence
- Each poet sees context
- Mimics human collaboration
- Prevents contradiction

**Trade-off**: Slower but higher quality

### 2. Structured Fact Extraction

**Chosen**: Pre-extract facts into categories

**Why**:
- Verifiable grounding
- Easier to reference
- Trackable (know which facts used)
- Judgeable (check accuracy)

**Trade-off**: Extra API call but worth it

### 3. Dual Poet Architecture

**Chosen**: Two poets with distinct styles

**Why**:
- Creates dialogue and tension
- More dynamic than single voice
- Not chaotic like 3+ poets
- Fair judging (equal opportunity)

**Trade-off**: Could add more poets, but two is sweet spot

### 4. Multi-Criteria Judging

**Chosen**: 5 weighted criteria + qualitative feedback

**Why**:
- Transparent (not black box)
- Actionable (know what to improve)
- Fair (multiple perspectives)
- Rich (captures complexity)

**Trade-off**: More complex but much better

### 5. Functional

**Chosen**: Pure functions + LangGraph state

**Why**:
- Simpler with LangChain patterns
- Easier to test
- Better composability
- No hidden state
- Aligns with modern Python trends

**Trade-off**: None really - functional is better here

---

## Learning Resources

### LangChain
- [LangChain Docs](https://python.langchain.com/)
- [LangChain Tutorials](https://python.langchain.com/docs/tutorials/)

### LangGraph
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

### Groq
- [Groq Documentation](https://console.groq.com/docs)
- [Groq Playground](https://console.groq.com/playground)

---
