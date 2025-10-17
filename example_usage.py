"""
Demonstrates various features of the Collaborative Poetry System
"""
import os
import json
import glob
from pathlib import Path
from poetry_system import generate_collaborative_poem, display_results
from audio_generator import generate_audio, GTTS_AVAILABLE, PYTTSX3_AVAILABLE

def check_setup():
    # Check API key
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        print(f"GROQ_API_KEY: Set ({api_key[:10]}...)")
    else:
        return False
    
    return True


def example_basic():
    """Example 1: Basic usage with text file"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Usage - Generate poem from text")
    print("="*80)
    
    # Create sample document
    sample_text = """The Wright Brothers, Orville and Wilbur, achieved the first powered flight 
        on December 17, 1903, in Kitty Hawk, North Carolina. Their aircraft, the Wright Flyer, 
        stayed airborne for 12 seconds and covered 120 feet. This historic moment marked 
        the beginning of modern aviation. The brothers had spent years studying bird flight 
        and experimenting with gliders before their breakthrough."""
        
    doc_path = "examples/wright_brothers.txt"
    os.makedirs("examples", exist_ok=True)
    
    with open(doc_path, "w") as f:
        f.write(sample_text)
    
    print(f"\nCreated document: {doc_path}")
    print(f"Content preview: {sample_text[:100]}...")
    
    # Generate poem
    print("\nGenerating poem...")
    results = generate_collaborative_poem(doc_path, num_verses=6)
    
    # Display
    display_results(results)
    
    # Save
    output_path = "examples/wright_brothers_poem.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to: {output_path}")
    
    return results

def example_comparison():
    """Example 2: Compare poems from different topics"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Topic Comparison - Science vs History")
    print("="*80)
    
    # Two different documents
    docs = {
        "science": """DNA, or deoxyribonucleic acid, is the hereditary material in humans and 
        almost all other organisms. James Watson and Francis Crick discovered the double helix 
        structure of DNA in 1953. DNA consists of two strands that coil around each other to 
        form a double helix carrying genetic instructions.""",
        
        "history": """The fall of the Berlin Wall on November 9, 1989, marked the end of the 
        Cold War era. The wall had divided East and West Berlin for 28 years. Its destruction 
        symbolized the reunification of Germany and the collapse of communist regimes across 
        Eastern Europe."""
    }
    
    os.makedirs("examples", exist_ok=True)
    
    results_all = {}
    
    for topic, content in docs.items():
        doc_path = f"examples/{topic}.txt"
        with open(doc_path, "w") as f:
            f.write(content)
        
        print(f"\n Topic: {topic.upper()}")
        print(f"Generating poem...")
        
        results = generate_collaborative_poem(doc_path, num_verses=6)
        results_all[topic] = results
        
        # Show winner
        print(f" Winner: {results['winner']}")
    
    # Compare
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    for topic, results in results_all.items():
        print(f"\n{topic.upper()}:")
        for judgment in results['judgments']:
            criteria = judgment['criteria']
            overall = sum(criteria.values()) / len(criteria)
            print(f"  {judgment['poet_name']}: {overall:.2f}/10")
    
    return results_all

def example_audio_output():
    """Example 3: Generate audio version"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Audio Output - Text to Speech")
    print("="*80)
    
    if not (GTTS_AVAILABLE or PYTTSX3_AVAILABLE):
        print("\nNo TTS engines available")
        print("Install with: pip install gTTS pyttsx3")
        return None
    
    # Create simple poem
    sample_text = """The Mona Lisa, painted by Leonardo da Vinci between 1503 and 1519, 
        is perhaps the most famous painting in the world. It hangs in the Louvre Museum 
        in Paris and attracts millions of visitors each year. The subject's enigmatic 
        smile has fascinated viewers for centuries."""
    
    doc_path = "examples/mona_lisa.txt"
    with open(doc_path, "w") as f:
        f.write(sample_text)
    
    print(f"\n Document: {doc_path}")
    print(" Generating poem...")
    
    results = generate_collaborative_poem(doc_path, num_verses=4)
    
    print("\n Generating audio...")
    
    # Try both engines
    audio_files = []
    
    if GTTS_AVAILABLE:
        print("  Using Google TTS...")
        audio_path = generate_audio(
            results['verses'],
            output_path="examples/poem_gtts.mp3",
            engine="gtts",
            include_intro=True
        )
        audio_files.append(audio_path)
    
    if PYTTSX3_AVAILABLE:
        print("  Using pyttsx3...")
        audio_path = generate_audio(
            results['verses'],
            output_path="examples/poem_pyttsx3.wav",
            engine="pyttsx3",
            include_intro=True
        )
        audio_files.append(audio_path)
    
    print(f"\n Generated {len(audio_files)} audio file(s)")
    for path in audio_files:
        print(f"{path}")
    
    return audio_files


def example_batch_processing():
    """Example 4: Process multiple documents"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Processing - Multiple documents")
    print("="*80)
    
    # Create multiple sample documents
    documents = {
        "tesla": """Nikola Tesla was a Serbian-American inventor born in 1856. He developed 
        the alternating current (AC) electrical system, which became the standard for power 
        transmission. Tesla held nearly 300 patents and made groundbreaking contributions 
        to wireless communication and electric motors.""",
                
        "everest": """Mount Everest, at 29,032 feet, is Earth's highest mountain. Edmund 
        Hillary and Tenzing Norgay first reached its summit on May 29, 1953. Located in the 
        Himalayas on the Nepal-Tibet border, Everest attracts hundreds of climbers annually.""",
                
        "internet": """The Internet began as ARPANET in 1969, a U.S. Defense Department 
        project. The first message was sent between UCLA and Stanford on October 29, 1969. 
        Tim Berners-Lee invented the World Wide Web in 1989, making the Internet accessible 
        to the public."""
    }
    
    os.makedirs("examples/batch", exist_ok=True)
    
    # Create documents
    for name, content in documents.items():
        with open(f"examples/batch/{name}.txt", "w") as f:
            f.write(content)
    
    print(f"\n Created {len(documents)} documents")
    
    # Process all
    results_batch = {}
    
    for doc_name in documents.keys():
        doc_path = f"examples/batch/{doc_name}.txt"
        print(f"\n Processing: {doc_name}")
        
        try:
            results = generate_collaborative_poem(doc_path, num_verses=4)
            results_batch[doc_name] = results
            
            # Save individual result
            output_path = f"examples/batch/{doc_name}_poem.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Winner: {results['winner']}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    
    print(f"\nProcessed: {len(results_batch)}/{len(documents)} documents")
    
    # Winner distribution
    winners = {}
    for doc_name, results in results_batch.items():
        winner = results['winner']
        winners[winner] = winners.get(winner, 0) + 1
    
    print("\nWinner distribution:")
    for poet, count in winners.items():
        print(f"  {poet}: {count} wins")
    
    return results_batch


def example_custom_length():
    """Example 5: Different poem lengths"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Lengths - Short vs Long poems")
    print("="*80)
    
    sample_text = """The Apollo 13 mission launched on April 11, 1970, but suffered an oxygen 
        tank explosion two days later. The famous phrase "Houston, we've had a problem" was 
        reported by astronaut Jack Swigert. Despite the crisis, the crew—Jim Lovell, Jack 
        Swigert, and Fred Haise—safely returned to Earth on April 17, 1970. The mission became 
        known as a "successful failure" demonstrating NASA's problem-solving capabilities."""
    
    doc_path = "examples/apollo13.txt"
    with open(doc_path, "w") as f:
        f.write(sample_text)
    
    lengths = [4, 8, 12]
    
    for length in lengths:
        print(f"\nGenerating {length}-verse poem...")
        results = generate_collaborative_poem(doc_path, num_verses=length)
        
        print(f"Generated {len(results['verses'])} verses")
        print(f"Winner: {results['winner']}")
        
        # Show first and last verse
        if results['verses']:
            print(f"First: \"{results['verses'][0]['line'][:50]}...\"")
            print(f"Last:  \"{results['verses'][-1]['line'][:50]}...\"")


def example_export_formats():
    """Example 6: Export to different formats"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Export Formats - JSON, Markdown, Plain Text")
    print("="*80)
    
    sample_text = """The Eiffel Tower was built for the 1889 World's Fair in Paris. 
        Designed by engineer Gustave Eiffel, it stands 330 meters tall. Initially criticized, 
        it has become the most-visited paid monument in the world, with nearly 7 million 
        visitors annually."""
    
    doc_path = "examples/eiffel.txt"
    with open(doc_path, "w") as f:
        f.write(sample_text)
    
    print("\nGenerating poem...")
    results = generate_collaborative_poem(doc_path, num_verses=6)
    
    os.makedirs("examples/exports", exist_ok=True)
    
    # Export JSON
    json_path = "examples/exports/poem.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Exported JSON: {json_path}")
    
    # Export Markdown
    md_path = "examples/exports/poem.md"
    with open(md_path, "w") as f:
        f.write("# Collaborative Poem\n\n")
        f.write("## The Poem\n\n")
        for verse in results['verses']:
            f.write(f"**Verse {verse['line_number']} - {verse['author']}**  \n")
            f.write(f"*{verse['line']}*\n\n")
        
        f.write("## Judgments\n\n")
        for judgment in results['judgments']:
            f.write(f"### {judgment['poet_name']}\n\n")
            overall = sum(judgment['criteria'].values()) / len(judgment['criteria'])
            f.write(f"**Overall Score:** {overall:.2f}/10\n\n")
    print(f"Exported Markdown: {md_path}")
    
    # Export Plain Text
    txt_path = "examples/exports/poem.txt"
    with open(txt_path, "w") as f:
        f.write("COLLABORATIVE POEM\n")
        f.write("="*60 + "\n\n")
        for verse in results['verses']:
            f.write(f"{verse['line_number']}. [{verse['author']}]\n")
            f.write(f"   {verse['line']}\n\n")
        
        f.write("\nWINNER: " + results['winner'] + "\n")
    print(f"Exported Plain Text: {txt_path}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("COLLABORATIVE AI POETRY SYSTEM - EXAMPLES".center(80))
    print("="*80)
    
    # Check setup first
    if not check_setup():
        print("\nSetup incomplete. Please configure GROQ_API_KEY.")
        return
    
    print("\nRunning examples...")
    print("(This will take a few minutes - Groq API calls)\n")
    
    examples = [
        ("Basic Usage", example_basic),
        ("Topic Comparison", example_comparison),
        ("Audio Output", example_audio_output),
        ("Batch Processing", example_batch_processing),
        ("Custom Lengths", example_custom_length),
        ("Export Formats", example_export_formats),
    ]
    
    completed = []
    failed = []
    
    for name, func in examples:
        try:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print(f"{'='*80}")
            func()
            completed.append(name)
        except Exception as e:
            print(f"\nError in {name}: {e}")
            failed.append(name)
    
    # Summary
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE".center(80))
    print("="*80)
    print(f"\nCompleted: {len(completed)}/{len(examples)}")
    for name in completed:
        print(f"{name}")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for name in failed:
            print(f"{name}")
    
    print(f"\nCheck the 'examples/' folder for all generated files")
    print("\nAll examples complete!")

if __name__ == "__main__":
    main()