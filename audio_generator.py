import os
import platform
from typing import List, Dict
import pyttsx3
from gtts import gTTS

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# CORE AUDIO GENERATION FUNCTIONS
def generate_audio_pyttsx3(verses: List[Dict], output_path: str = "poem.wav",include_intro: bool = True) -> str:
    
    engine = pyttsx3.init()
    
    # Build text
    text_parts = []
    
    if include_intro:
        text_parts.append("A Collaborative Poem.")
        text_parts.append("Created by two AI poets working together.")
        text_parts.append("")  # Pause
    
    for verse in verses:
        text_parts.append(f"Verse {verse['line_number']}.")
        text_parts.append(verse['line'])
        text_parts.append("")  # Pause
    
    full_text = " ".join(text_parts)
    
    # Configure voice
    engine.setProperty('rate', 150)  # Slower for poetry
    engine.setProperty('volume', 0.9)
    
    # Generate audio
    engine.save_to_file(full_text, output_path)
    engine.runAndWait()
    
    print(f"Audio saved to {output_path}")
    return output_path


def generate_audio_gtts(verses: List[Dict],
                       output_path: str = "poem.mp3",
                       include_intro: bool = True,
                       slow: bool = True) -> str:
    if not GTTS_AVAILABLE:
        raise ImportError("gTTS not installed. Run: pip install gTTS")
    
    # Build text
    text_parts = []
    
    if include_intro:
        text_parts.append("A Collaborative Poem.")
        text_parts.append("")
    
    for verse in verses:
        text_parts.append(f"Verse {verse['line_number']}.")
        text_parts.append(verse['line'])
        text_parts.append("")  # Pause
    
    full_text = " ".join(text_parts)
    
    # Generate audio
    tts = gTTS(text=full_text, lang='en', slow=slow)
    tts.save(output_path)
    
    print(f"Audio saved to {output_path}")
    return output_path


def generate_audio(verses: List[Dict],
                  output_path: str = None,
                  engine: str = "auto",
                  include_intro: bool = True) -> str:

    # Auto-select engine
    if engine == "auto":
        if GTTS_AVAILABLE:
            engine = "gtts"
        elif PYTTSX3_AVAILABLE:
            engine = "pyttsx3"
    
    # Auto-determine output path
    if output_path is None:
        ext = "mp3" if engine == "gtts" else "wav"
        output_path = f"poem_audio.{ext}"
    
    # Generate using selected engine
    if engine == "pyttsx3":
        return generate_audio_pyttsx3(verses, output_path, include_intro)
    elif engine == "gtts":
        return generate_audio_gtts(verses, output_path, include_intro)
    else:
        raise ValueError(f"Unknown engine: {engine}")

def play_audio(audio_path: str) -> bool:

    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            os.system(f"afplay '{audio_path}'")
        elif system == "Linux":
            # Try different players
            if os.system(f"which mpg123 > /dev/null 2>&1") == 0:
                os.system(f"mpg123 '{audio_path}'")
            elif os.system(f"which aplay > /dev/null 2>&1") == 0:
                os.system(f"aplay '{audio_path}'")
            else:
                print(f"Audio saved to {audio_path}")
                print("Install mpg123 or aplay to play audio")
                return False
        elif system == "Windows":
            os.system(f"start '{audio_path}'")
        else:
            print(f"Audio saved to {audio_path}")
            print("Platform not recognized - please play manually")
            return False
        
        return True
        
    except Exception as e:
        print(f"Could not play audio: {e}")
        print(f"Audio saved to {audio_path} - please play manually")
        return False

# HELPER FUNCTIONS
def format_poem_for_audio(verses: List[Dict], 
                         include_authors: bool = False) -> str:

    lines = []
    
    for verse in verses:
        if include_authors:
            lines.append(f"From {verse['author']}:")
        
        lines.append(f"Verse {verse['line_number']}.")
        lines.append(verse['line'])
        lines.append("")  # Pause
    
    return " ".join(lines)

def get_available_engines() -> List[str]:

    engines = []
    
    if PYTTSX3_AVAILABLE:
        engines.append("pyttsx3")
    if GTTS_AVAILABLE:
        engines.append("gtts")
    
    return engines


def check_audio_capability() -> Dict:

    return {
        'pyttsx3_available': PYTTSX3_AVAILABLE,
        'gtts_available': GTTS_AVAILABLE,
        'platform': platform.system(),
        'available_engines': get_available_engines()
    }

# EXAMPLE USAGE
def main():
    """Example usage"""
    
    # Check capabilities
    capabilities = check_audio_capability()
    print("Audio Generator")
    print("="*60)
    print(f"Platform: {capabilities['platform']}")
    print(f"pyttsx3: {'Found' if capabilities['pyttsx3_available'] else 'none'}")
    print(f"gTTS: {'Found' if capabilities['gtts_available'] else 'none'}")
    print()
    
    if not capabilities['available_engines']:
        print("No TTS engines available!")
        print("Install with: pip install gTTS pyttsx3")
        return
    
    # Sample verses
    sample_verses = [
        {
            'line': "Like eagles breaking dawn's silence, two souls pierced the lunar veil.",
            'author': "Poet A (Metaphorical)",
            'line_number': 1,
            'factual_anchors': ["Apollo 11", "moon landing"]
        },
        {
            'line': "Armstrong's boot marked July 20, 1969, in dust forever still.",
            'author': "Poet B (Narrative)",
            'line_number': 2,
            'factual_anchors': ["Neil Armstrong", "July 20, 1969"]
        },
        {
            'line': "Six hundred million hearts beat as one beneath Earth's fragile sphere.",
            'author': "Poet A (Metaphorical)",
            'line_number': 3,
            'factual_anchors': ["600 million viewers"]
        },
        {
            'line': "They collected forty-seven pounds of ancient rock to bring home to science.",
            'author': "Poet B (Narrative)",
            'line_number': 4,
            'factual_anchors': ["47.5 pounds lunar material"]
        },
    ]
    
    # Generate audio with each available engine
    for engine in capabilities['available_engines']:
        print(f"\nGenerating with {engine}...")
        try:
            audio_path = generate_audio(
                sample_verses,
                engine=engine,
                include_intro=True
            )
            print(f"Success: {audio_path}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Audio generation complete!")
    print("Play the generated files to hear your poem!")

if __name__ == "__main__":
    main()