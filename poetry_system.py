### STATE DEFINITIONS
from typing import TypedDict
from typing import Dict
from typing import Annotated
from typing import List
import operator
from pathlib import Path
from PIL import Image
# import pytesseract
import PyPDF2
from docx import Document as DocxDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import json

load_dotenv()


class PoemState(TypedDict):
    """State that flows through the LangGraph workflow"""
    document_path: str
    document_text: str
    factual_context: Dict
    verses: Annotated[List[Dict], operator.add]
    current_verse_num: int
    max_verses: int
    poet_turn: str  # "poet_a" or "poet_b"
    judgments: List[Dict]
    winner: str
    error: str

### Document processing
def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF"""
    try:
        text = []
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX"""
    try:
        doc = DocxDocument(docx_path)
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_txt(txt_path: str) -> str:
    """Read text file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error: {str(e)}"

def process_document(file_path: str) -> str:
    """Route document to appropriate processor"""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    processors = {
        '.jpg': extract_text_from_image,
        '.jpeg': extract_text_from_image,
        '.png': extract_text_from_image,
        '.pdf': extract_text_from_pdf,
        '.docx': extract_text_from_docx,
        '.doc': extract_text_from_docx,
        '.txt': extract_text_from_txt
    }
    
    processor = processors.get(ext)
    if processor:
        return processor(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

### CHAINS FOR EACH TASK
def create_fact_extraction_chain(llm):
    """Create LangChain for extracting structured facts"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a fact extraction expert. Extract structured information from text.
        Return ONLY a valid JSON object with these exact keys:
        - entities: list of people, places, organizations, concepts
        - key_facts: list of concrete factual statements
        - themes: list of main themes or topics
        - temporal_info: list of dates, time periods, sequences
        - numerical_data: list of numbers, statistics, measurements

        Be precise and extract actual facts from the text."""),
        ("human", "Extract facts from this text:\n\n{text}\n\nReturn JSON only:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain


def create_poet_chain(llm, poet_name: str, style: str):
    """Create LangChain for a specific poet"""
    
    personas = {
        "metaphorical": """You are a poet who transforms facts into vivid metaphors and imagery.
        You MUST reference specific facts but express them through symbolism, nature imagery, and sensory details.
        You maintain strict factual accuracy while elevating language to art.""",
        
        "narrative": """You are a poet who tells clear stories with concrete details.
        You MUST reference specific facts directly - names, dates, numbers, events.
        You craft verses with clarity, accessibility, and emotional resonance."""
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", personas[style]),
        ("human", """CRITICAL INSTRUCTIONS:
        1. You MUST reference at least ONE specific fact from the context below
        2. Use actual names, dates, numbers, or events from the context
        3. Do NOT make up information
        4. Continue naturally from previous verses

        FACTUAL CONTEXT (USE THESE FACTS):
        {context}

        POEM SO FAR:
        {previous_verses}

        Now write verse #{verse_num} in {style} style.

        Format your response EXACTLY like this:
        LINE: [your verse here - 1-2 lines, max 20 words]
        FACTS_USED: [comma-separated list of specific facts you referenced]

        Example:
        LINE: Armstrong stepped on lunar soil, July 20, marking history forever.
        FACTS_USED: Neil Armstrong, July 20 1969, moon landing""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain

def create_judging_chain(llm):
    """Create LangChain for judging poetry"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert poetry critic. Evaluate contributions fairly and provide specific feedback."""),
        ("human", """Evaluate {poet_name}'s verses in this collaborative poem.

        FACTUAL SOURCE MATERIAL:
        {context}

        FULL POEM:
        {full_poem}

        {poet_name}'s VERSES ONLY:
        {poet_verses}

        Rate each criterion 0-10:
        1. FACTUAL_ACCURACY: How well grounded in source material? Do verses reference actual facts?
        2. POETIC_QUALITY: Aesthetic merit, imagery, rhythm, word choice
        3. COHERENCE: Flow with overall poem and previous verses
        4. CREATIVITY: Originality, fresh perspectives, unexpected connections
        5. EMOTIONAL_IMPACT: Evocative power, ability to engage reader

        Respond EXACTLY in this format:

        SCORES:
        factual_accuracy: [number 0-10]
        poetic_quality: [number 0-10]
        coherence: [number 0-10]
        creativity: [number 0-10]
        emotional_impact: [number 0-10]

        STRENGTHS:
        - [specific strength with example]
        - [specific strength with example]
        - [specific strength with example]

        WEAKNESSES:
        - [specific weakness with example]
        - [specific weakness with example]

        STANDOUT_VERSES: [comma-separated line numbers, e.g. 1,3,7]

        REASONING:
        [2-3 paragraphs explaining your scores with specific examples from their verses]""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain

### LANGGRAPH NODE FUNCTIONS
def load_document_node(state: PoemState) -> Dict:
    """Node: Load and process document"""
    try:
        text = process_document(state['document_path'])
        return {"document_text": text}
    except Exception as e:
        return {"error": f"Document processing failed: {str(e)}"}

def extract_facts_node(state: PoemState) -> Dict:
    """Node: Extract factual context using LangChain"""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    
    chain = create_fact_extraction_chain(llm)
    
    try:
        result = chain.invoke({"text": state['document_text']})
        
        # Parse JSON from response
        result = result.strip()
        if result.startswith('```json'):
            result = result.split('```json')[1].split('```')[0].strip()
        elif result.startswith('```'):
            result = result.split('```')[1].split('```')[0].strip()
        
        facts = json.loads(result)
        
        print(f"   Entities: {len(facts.get('entities', []))}")
        print(f"   Key Facts: {len(facts.get('key_facts', []))}")
        print(f"   Themes: {len(facts.get('themes', []))}")
        
        return {"factual_context": facts}
        
    except Exception as e:
        print(f"   Fact extraction error: {e}")
        # Fallback: create basic context
        return {"factual_context": {
            "entities": ["subject from document"],
            "key_facts": ["main event or information"],
            "themes": ["primary theme"],
            "temporal_info": ["timeframe if mentioned"],
            "numerical_data": ["statistics if present"]
        }}

def generate_verse_node(state: PoemState) -> Dict:
    """Node: Generate next verse"""
    verse_num = state['current_verse_num']
    poet_turn = state['poet_turn']
    
    # Determine poet details
    if poet_turn == "poet_a":
        poet_name = "Poet A (Metaphorical)"
        style = "metaphorical"
    else:
        poet_name = "Poet B (Narrative)"
        style = "narrative"
    
    print(f"  Verse {verse_num}: {poet_name}")
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    
    chain = create_poet_chain(llm, poet_name, style)
    
    # Format previous verses
    if state['verses']:
        previous_verses = "\n".join([
            f"{v['line_number']}. [{v['author']}] {v['line']}"
            for v in state['verses']
        ])
    else:
        previous_verses = "[This is the first verse of the poem]"
    
    # Format context - CRITICAL FIX
    fc = state['factual_context']
    context_parts = []
    
    if fc.get('entities'):
        context_parts.append(f"ENTITIES: {', '.join(fc['entities'][:8])}")
    if fc.get('key_facts'):
        context_parts.append(f"KEY FACTS: {' | '.join(fc['key_facts'][:5])}")
    if fc.get('themes'):
        context_parts.append(f"THEMES: {', '.join(fc['themes'][:4])}")
    if fc.get('temporal_info'):
        context_parts.append(f"DATES/TIME: {', '.join(fc['temporal_info'][:4])}")
    if fc.get('numerical_data'):
        context_parts.append(f"NUMBERS: {', '.join(map(str, fc['numerical_data'][:4]))}")
    
    context = "\n".join(context_parts) if context_parts else "No specific facts extracted"
    
    try:
        result = chain.invoke({
            "context": context,
            "previous_verses": previous_verses,
            "verse_num": verse_num,
            "style": style
        })
        
        # Parse result
        lines = result.strip().split('\n')
        verse_line = ""
        facts_used = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('LINE:'):
                verse_line = line.replace('LINE:', '').strip()
            elif line.startswith('FACTS_USED:'):
                facts_str = line.replace('FACTS_USED:', '').strip()
                facts_used = [f.strip() for f in facts_str.split(',') if f.strip()]
        
        # Fallback if parsing fails
        if not verse_line:
            verse_line = result.strip().split('\n')[0].replace('LINE:', '').strip()
        
        verse = {
            "line": verse_line,
            "author": poet_name,
            "line_number": verse_num,
            "factual_anchors": facts_used
        }
        
        print(f"   \"{verse_line[:60]}...\"")
        
        # Update state
        next_turn = "poet_b" if poet_turn == "poet_a" else "poet_a"
        
        return {
            "verses": [verse],
            "current_verse_num": verse_num + 1,
            "poet_turn": next_turn
        }
        
    except Exception as e:
        print(f"   Error: {e}")
        # Return placeholder verse
        verse = {
            "line": f"[Verse {verse_num} generation failed]",
            "author": poet_name,
            "line_number": verse_num,
            "factual_anchors": []
        }
        next_turn = "poet_b" if poet_turn == "poet_a" else "poet_a"
        return {
            "verses": [verse],
            "current_verse_num": verse_num + 1,
            "poet_turn": next_turn
        }

def check_completion_node(state: PoemState) -> str:
    """Node: Check if poem is complete"""
    if state['current_verse_num'] > state['max_verses']:
        return "judge"
    else:
        return "generate"

def judge_poems_node(state: PoemState) -> Dict:
    """Node: Judge both poets' contributions"""
    print("Judging contributions...")
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3,
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    
    chain = create_judging_chain(llm)
    
    # Separate verses by author
    poet_verses = {}
    for verse in state['verses']:
        author = verse['author']
        if author not in poet_verses:
            poet_verses[author] = []
        poet_verses[author].append(verse)
    
    # Format full poem
    full_poem = "\n".join([
        f"{v['line_number']}. [{v['author']}] {v['line']}"
        for v in state['verses']
    ])
    
    # Format context
    fc = state['factual_context']
    context = f"""Entities: {', '.join(fc.get('entities', [])[:8])}
        Key Facts: {'; '.join(fc.get('key_facts', [])[:5])}
        Themes: {', '.join(fc.get('themes', [])[:4])}
        Dates: {', '.join(fc.get('temporal_info', [])[:4])}
        Numbers: {', '.join(map(str, fc.get('numerical_data', [])[:4]))}
        """
    
    judgments = []
    
    for poet_name, verses in poet_verses.items():
        poet_lines = "\n".join([
            f"{v['line_number']}. {v['line']}"
            for v in verses
        ])
        
        try:
            print(f"   Evaluating {poet_name}...")
            result = chain.invoke({
                "poet_name": poet_name,
                "context": context,
                "full_poem": full_poem,
                "poet_verses": poet_lines
            })
            
            # Parse judgment
            judgment = parse_judgment(result, poet_name)
            judgments.append(judgment)
            
        except Exception as e:
            print(f"  Error judging {poet_name}: {e}")
            # Fallback judgment
            judgments.append({
                'poet_name': poet_name,
                'criteria': {
                    'factual_accuracy': 5.0,
                    'poetic_quality': 5.0,
                    'coherence': 5.0,
                    'creativity': 5.0,
                    'emotional_impact': 5.0
                },
                'strengths': ["Unable to evaluate - see error"],
                'weaknesses': ["Evaluation failed"],
                'standout_verses': [],
                'reasoning': f"Error during judgment: {str(e)}"
            })
    
    # Determine winner
    if len(judgments) == 2:
        score1 = calculate_overall_score(judgments[0]['criteria'])
        score2 = calculate_overall_score(judgments[1]['criteria'])
        winner = judgments[0]['poet_name'] if score1 > score2 else judgments[1]['poet_name']
    else:
        winner = "Unknown"
    
    return {
        "judgments": judgments,
        "winner": winner
    }

### HELPER FUNCTIONS
def parse_judgment(response: str, poet_name: str) -> Dict:
    """Parse judgment response into structured data"""
    lines = response.split('\n')
    
    scores = {}
    strengths = []
    weaknesses = []
    standout_verses = []
    reasoning = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith('SCORES:'):
            current_section = 'scores'
        elif line.startswith('STRENGTHS:'):
            current_section = 'strengths'
        elif line.startswith('WEAKNESSES:'):
            current_section = 'weaknesses'
        elif line.startswith('STANDOUT_VERSES:'):
            current_section = 'standout'
            verses_str = line.replace('STANDOUT_VERSES:', '').strip()
            try:
                standout_verses = [int(v.strip()) for v in verses_str.split(',') if v.strip().isdigit()]
            except:
                pass
        elif line.startswith('REASONING:'):
            current_section = 'reasoning'
        elif current_section == 'scores' and ':' in line:
            key, value = line.split(':', 1)
            try:
                scores[key.strip()] = float(value.strip())
            except ValueError:
                pass
        elif current_section == 'strengths' and line.startswith('-'):
            strengths.append(line[1:].strip())
        elif current_section == 'weaknesses' and line.startswith('-'):
            weaknesses.append(line[1:].strip())
        elif current_section == 'reasoning' and not line.startswith('-'):
            reasoning.append(line)
    
    return {
        'poet_name': poet_name,
        'criteria': {
            'factual_accuracy': scores.get('factual_accuracy', 5.0),
            'poetic_quality': scores.get('poetic_quality', 5.0),
            'coherence': scores.get('coherence', 5.0),
            'creativity': scores.get('creativity', 5.0),
            'emotional_impact': scores.get('emotional_impact', 5.0)
        },
        'strengths': strengths if strengths else ["Evaluation completed"],
        'weaknesses': weaknesses if weaknesses else ["See detailed feedback"],
        'standout_verses': standout_verses,
        'reasoning': '\n'.join(reasoning) if reasoning else "Detailed analysis provided above."
    }

def calculate_overall_score(criteria: Dict) -> float:
    """Calculate weighted overall score"""
    weights = {
        'factual_accuracy': 0.25,
        'poetic_quality': 0.25,
        'coherence': 0.20,
        'creativity': 0.15,
        'emotional_impact': 0.15
    }
    
    return sum(criteria[k] * weights[k] for k in weights.keys())

def display_results(results: Dict):
    """Pretty print results"""
    print("\n" + "="*80)
    print("COLLABORATIVE POEM".center(80))
    print("="*80 + "\n")
    
    for verse in results['verses']:
        print(f"{verse['line_number']}. [{verse['author']}]")
        print(f"   {verse['line']}")
        if verse.get('factual_anchors'):
            print(f"   Facts: {', '.join(verse['factual_anchors'][:3])}")
        print()
    
    print("\n" + "="*80)
    print("JUDGMENTS".center(80))
    print("="*80 + "\n")
    
    for judgment in results['judgments']:
        overall = calculate_overall_score(judgment['criteria'])
        
        print(f"\n{judgment['poet_name']}")
        print("-" * 40)
        print(f"Overall Score: {overall:.2f}/10")
        print(f"\nDetailed Scores:")
        for criterion, score in judgment['criteria'].items():
            print(f"  {criterion.replace('_', ' ').title()}: {score}/10")
        
        print(f"\nStrengths:")
        for strength in judgment['strengths']:
            print(f"  {strength}")
        
        print(f"\nWeaknesses:")
        for weakness in judgment['weaknesses']:
            print(f"  {weakness}")
        
        if judgment['standout_verses']:
            print(f"\nStandout Verses: {', '.join(map(str, judgment['standout_verses']))}")
        
        if judgment['reasoning']:
            print(f"\nReasoning:\n{judgment['reasoning'][:300]}...")
    
    print("\n" + "="*80)
    print(f"WINNER: {results['winner']}")
    print("="*80 + "\n")

### LANGGRAPH WORKFLOW
def create_poetry_workflow():
    """Create LangGraph workflow for collaborative poetry"""
    
    workflow = StateGraph(PoemState)
    
    # Add nodes
    workflow.add_node("load_document", load_document_node)
    workflow.add_node("extract_facts", extract_facts_node)
    workflow.add_node("generate_verse", generate_verse_node)
    workflow.add_node("judge_poems", judge_poems_node)
    
    # Define edges
    workflow.set_entry_point("load_document")
    workflow.add_edge("load_document", "extract_facts")
    workflow.add_edge("extract_facts", "generate_verse")
    
    # Conditional edge
    workflow.add_conditional_edges(
        "generate_verse",
        check_completion_node,
        {
            "generate": "generate_verse",
            "judge": "judge_poems"
        }
    )
    
    workflow.add_edge("judge_poems", END)
    
    return workflow.compile()

### MAIN FUNCTION
def generate_collaborative_poem(document_path: str, num_verses: int = 8) -> Dict:
    """Main function to generate collaborative poem"""
    
    # Create workflow
    app = create_poetry_workflow()
    
    # Initial state
    initial_state = {
        "document_path": document_path,
        "document_text": "",
        "factual_context": {},
        "verses": [],
        "current_verse_num": 1,
        "max_verses": num_verses,
        "poet_turn": "poet_a",
        "judgments": [],
        "winner": "",
        "error": ""
    }
    
    # Run workflow
    final_state = app.invoke(initial_state)
    
    return {
        'document_path': document_path,
        'factual_context': final_state['factual_context'],
        'verses': final_state['verses'],
        'judgments': final_state['judgments'],
        'winner': final_state['winner']
    }

def main():
    """Example usage"""
    
    # Check for API key
    api_key = os.getenv('GROQ_API_KEY')
    
    # Create sample document
    document_path = "./sample_document.txt"
    # document_path = "C:/Users/emon1/Desktop/2.0/task/sample_document.txt"
    
    if not os.path.exists(document_path):
        sample_text = """The Moon landing occurred on July 20, 1969, when Neil Armstrong 
        and Buzz Aldrin became the first humans to walk on the lunar surface. The Apollo 11 
        mission launched from Kennedy Space Center on July 16, 1969. Armstrong's famous words 
        "That's one small step for man, one giant leap for mankind" were broadcast to an 
        estimated 600 million people watching on Earth. The astronauts spent 21.5 hours on 
        the Moon's surface and collected 47.5 pounds of lunar material."""
        
        with open(document_path, 'w') as f:
            f.write(sample_text)
        print(f"Created sample document: {document_path}")
        
    
    # Generate poem
    print("\nStarting collaborative poetry generation...")
    print("="*80)
    
    results = generate_collaborative_poem(document_path, num_verses=8)
    
    # Display results
    display_results(results)
    
    # Save results
    output_file = 'poem_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
