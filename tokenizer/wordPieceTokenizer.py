import re
import requests
from collections import defaultdict, Counter

class WordPieceTokenizer():
    def __init__(self, vocab_size=10000):
        self.vocab = {}
        self.word_freqs = {}
        self.vocab_size = vocab_size
        self.inv_vocab = {}
        self.unk_token = "[UNK]"
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.wordpieces_prefix ="##"


   
    def fit(self, text):
        # Count word frequencies
        words = re.findall(r'\w+|[^\w\s]', text)
        self.word_freqs = Counter(words)

        alphabet = []
        for word in self.word_freqs.keys():
            # Add the first letter of the word to the alphabet if not exists
            if word[0] not in alphabet:
                alphabet.append(word[0])
            # Add the rest of the letters to the alphabet if not exist with a prefix
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        alphabet.sort()
        
        # Add special tokens to the vocabulary plus the created alphabet
        self.vocab = [self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token ] + alphabet.copy()
        # Create a dictionary with all words and all splitted characters
        splits = {
            word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)]
            for word in self.word_freqs.keys()
        }
        
        while len(self.vocab) < self.vocab_size:
            scores = self._compute_pair_scores(splits)
            if not scores:
                break
            best_pair, max_score = "", None
            for pair, score in scores.items():
                if max_score is None or max_score < score:
                    best_pair = pair
                    max_score = score
            
            splits = self._merge_pair(*best_pair, splits)
            new_token = (
                best_pair[0] + best_pair[1][2:]
                if best_pair[1].startswith("##")
                else best_pair[0] + best_pair[1]
            )
            self.vocab.append(new_token)
    
    def encode_word(self, word):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def encode_text(self, text):
        words = re.findall(r'\w+|[^\w\s]', text)
        tokens = []
        for word in words:
            tokens.extend(self.encode_word(word))
        return tokens
    
    def decode(self, tokens):
        return "".join(tokens).replace("##", "")
        
    def _compute_pair_scores(self, splits):
        letter_freqs = defaultdict(int)
        pair_freqs = defaultdict(int)
        # Compute the frequency of each letter and pair of consecutive letters
        for word, freq in self.word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                letter_freqs[split[0]] += freq
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                letter_freqs[split[i]] += freq
                pair_freqs[pair] += freq
            letter_freqs[split[-1]] += freq

        # Compute the score of each pair (pair frequency / (letter1 frequency * letter2 frequency)
        scores = {
            pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
            for pair, freq in pair_freqs.items()
        }
        return scores
    
    def _merge_pair(self, a, b, splits):
        for word in self.word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    merge = a + b[2:] if b.startswith("##") else a + b
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits


def load_separate_and_clean_stories(filename):
    with open(filename, 'r') as file:
        content = file.read()

    stories = content.split('\n\n\n\n')

    cleaned_stories = []
    for story in stories:
        cleaned_story = re.sub(r'\n\s*\n', '\n', story.strip())
        cleaned_stories.append(cleaned_story)
    
    return cleaned_stories


url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
response = requests.get(url)
text = response.text

start = text.find('Chapter I.]')
end = text.rfind('END OF THE PROJECT GUTENBERG EBOOK')
text = text[start:end]


corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]
corpus = " ".join(corpus)
# Train the tokenizer
tokenizer = WordPieceTokenizer(1000)
tokenizer.fit(corpus)
print(tokenizer.encode_word("Hugging"))
print(tokenizer.decode("H0gging"))
