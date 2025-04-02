import json
import re
from pathlib import Path
from tqdm import tqdm

class MetadataOptimizer:
    def __init__(self):
        # Extended hair color keywords
        self.hair_keywords = {
            'black': ['black', 'dark', 'ink', 'jet', 'onyx', 'raven', 'sable', 'shadow', 'ebony', 'pitch', 'noir'],
            'blonde': ['blonde', 'golden', 'strawberry', 'platinum', 'honey', 'sandy', 'flaxen', 'butterscotch', 'champagne', 'sun-kissed', 'wheat', 'amber'],
            'brown': ['brown', 'chestnut', 'auburn', 'mahogany', 'walnut', 'caramel', 'cinnamon', 'copper', 'hazel', 'tawny', 'umber'],
            'red': ['red', 'crimson', 'ruby', 'scarlet', 'burgundy', 'vermilion', 'rust', 'cardinal', 'maroon', 'garnet', 'cerise'],
            'white': ['white', 'silver', 'platinum', 'ivory', 'alabaster', 'pearl', 'ash', 'frost', 'moonlight', 'ghost', 'pale', 'bleached'],
            'blue': ['blue', 'azure', 'cobalt', 'sapphire', 'navy', 'teal', 'cyan', 'cerulean', 'indigo', 'aqua', 'sky'],
            'pink': ['pink', 'rose', 'coral', 'fuchsia', 'hot pink', 'pastel', 'bubblegum', 'salmon', 'blush', 'magenta', 'raspberry'],
            'green': ['green', 'emerald', 'lime', 'olive', 'jade', 'mint', 'forest', 'chartreuse', 'sage', 'sea', 'army'],
            'purple': ['purple', 'lavender', 'violet', 'amethyst', 'lilac', 'plum', 'orchid', 'mauve', 'grape', 'royal', 'byzantium'],
            'others': []
        }
        
        # Extended personality keywords
        self.personality_keywords = {
            'cheerful': ['cheerful', 'optimistic', 'jovial', 'vivacious', 'bright', 'radiant', 'bubbly', 'perky', 'sunny', 'lighthearted', 'exuberant'],
            'aggressive': ['aggressive', 'hostile', 'combative', 'assertive', 'domineering', 'confrontational', 'forceful', 'intense', 'quarrelsome', 'turbulent', 'volatile'],
            'arrogant': ['arrogant', 'haughty', 'conceited', 'smug', 'supercilious', 'pompous', 'egotistical', 'cocky', 'snobbish', 'imperious', 'overbearing'],
            'shy': ['shy', 'timid', 'bashful', 'reserved', 'introverted', 'reticent', 'withdrawn', 'coy', 'demure', 'self-conscious', 'mousy'],
            'loyal': ['loyal', 'faithful', 'devoted', 'steadfast', 'trustworthy', 'dependable', 'committed', 'dedicated', 'staunch', 'true-blue', 'unalterable'],
            'mysterious': ['mysterious', 'enigmatic', 'cryptic', 'inscrutable', 'occult', 'arcane', 'puzzling', 'unfathomable', 'sphinxlike', 'uncanny', 'obscure'],
            'calm': ['calm', 'composed', 'serene', 'tranquil', 'placid', 'unflappable', 'poised', 'collected', 'level-headed', 'imperturbable', 'stoic'],
            'genius': ['genius', 'prodigy', 'brilliant', 'gifted', 'astute', 'ingenious', 'intellectual', 'savant', 'whiz', 'polymath', 'virtuoso'],
            'clumsy': ['clumsy', 'awkward', 'bungling', 'gauche', 'maladroit', 'inept', 'uncoordinated', 'klutzy', 'heavy-handed', 'butterfingers', 'ham-fisted'],
            'others': []
        }

    def process_file(self, input_path, output_path):
        """Process metadata file"""
        with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
            for line in tqdm(fin, desc="Processing metadata"):
                data = json.loads(line)
                optimized = self._optimize_record(data)
                fout.write(json.dumps(optimized) + '\n')

    def _optimize_record(self, data):
        """Optimize a single record"""
        # Extract UUID as unique ID
        file_name = Path(data['file_name']).stem
        record_id = file_name.split('-')[0]
        
        # Structured processing
        return {
            "id": record_id,
            "hair": self._extract_hair_color(data['appearance']),
            "personality": self._extract_personality(data.get('personality', '')),
            "appearance_keys": self._extract_appearance_keys(data['appearance'])
        }

    def _extract_hair_color(self, text):
        """Intelligently extract hair color"""
        text = text.lower()
        for color, keywords in self.hair_keywords.items():
            if any(re.search(r'\b' + kw + r'\b', text) for kw in keywords):
                return color
        return 'others'

    def _extract_personality(self, text):
        """Extract structured personality traits"""
        if not text:
            return []
            
        text = text.lower()
        traits = []
        for trait, keywords in self.personality_keywords.items():
            if any(re.search(r'\b' + kw + r'\b', text) for kw in keywords):
                traits.append(trait)
        return traits

    def _extract_appearance_keys(self, text):
        """Extract key appearance features"""
        # Remove useless descriptions
        text = re.sub(r'$$.*?$$|\d+', '', text)  # Remove reference tags and numbers
        text = text.lower()
        
        # Extract core noun phrases
        keywords = []
        patterns = [
            r'\b(\w+ed) \w+\b',    # Capture adjective + noun structure
            r'\b\w+ (hair|eyes)\b', # Capture body part descriptions
            r'\b\w+[- ]\w+\b'       # Capture compound words
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                keywords.extend(match.split())
        
        # Deduplication and normalization
        return list(set([
            re.sub(r's$', '', kw)  # Singularize
            for kw in keywords if len(kw) > 3  # Filter short words
        ]))

if __name__ == "__main__":
    optimizer = MetadataOptimizer()
    optimizer.process_file(
        input_path="/home/xyc/ML/data/AniPersonaCaps/metadata.jsonl",
        output_path="/home/xyc/ML/CLIP/optimized_metadata.jsonl"
    )