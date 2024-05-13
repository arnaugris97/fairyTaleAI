import json
import re
import urllib.request
from collections import defaultdict, Counter

class WordPieceTokenizer():
    def __init__(self, vocab_size=10000):
        self.vocab = {}
        self.word_freqs = {}
        self.vocab_size = vocab_size
        self.unk_token = "[UNK]"
        self.aps_token = "[APS]"
        self.space_token = "[SPACE]"
        self.brk_token = "[BRK]"
        self.sep_token = "[SEP]"
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.mask_token = "[MASK]"
        self.wordpieces_prefix ="##"

   
    def fit(self, text):
        # Count word frequencies
        text = re.sub(r'\n+', ' ' + self.brk_token + ' ', text)
        text = re.sub(r"\s'\s", self.space_token + self.aps_token + self.space_token, text)
        text = re.sub(r"\s'", self.space_token + self.aps_token, text)
        text = re.sub(r"'\s",  self.aps_token + self.space_token, text)
        # Change charcater ' to [APS]
        text = re.sub(r'\'', self.aps_token, text)
        words = re.findall(r'\w+[\w.,;!?\'\"-]*|[\.,;!?\'\"-]+', text)
        
        self.word_freqs = Counter(words)

        alphabet = []
        for word in self.word_freqs.keys():
            if word == self.brk_token or word == self.aps_token or word == self.space_token:
                continue 
            # Add the first letter of the word to the alphabet if not exists
            if word[0] not in alphabet:
                alphabet.append(word[0])
            # Add the rest of the letters to the alphabet if not exist with a prefix
            for letter in word[1:]:
                if f"##{letter}" not in alphabet:
                    alphabet.append(f"##{letter}")

        alphabet.sort()
        
        # Add special tokens to the vocabulary plus the created alphabet
        self.vocab = [self.unk_token, self.cls_token, self.sep_token, self.space_token, self.pad_token, self.mask_token, self.brk_token, self.aps_token ] + alphabet.copy()
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
        print(self.vocab)
        print(len(self.vocab))
        print(self.word_freqs)
    
    
    def encode(self, text):
        # Normalize and split the text
        text = re.sub(r'\n+', ' ' + self.brk_token + ' ', text)
        text = re.sub(r"\s'\s", self.space_token + self.aps_token + self.space_token, text)
        text = re.sub(r"\s'", self.space_token + self.aps_token, text)
        text = re.sub(r"'\s",  self.aps_token + self.space_token, text)
        # Change charcater ' to [APS] and
        text = re.sub(r'\'', self.aps_token, text)
        pattern = r'\w+[\w.,;!?\'\"-]*|[\.,;!?\'\"-]+|(?:' + re.escape(self.brk_token) + r'|' + re.escape(self.aps_token) + r'|' + re.escape(self.space_token) + r')'
        words = re.findall(pattern, text)
        
        # Tokenize into words and subwords
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                sub_tokens = self.tokenize_word(word)
                tokens.extend(sub_tokens)

        return tokens

    def tokenize_word(self, word):
        if word == self.brk_token:
            return [self.brk_token]
        if word == self.aps_token:
            return [self.aps_token]
        if word == self.space_token:
            return [self.space_token]
        
        subwords = []
        start = 0
        while start < len(word):
            match = False
            for end in range(len(word), start, -1):
                subword = word[start:end]
                if start > 0:
                    subword = "##" + subword
                if subword in self.vocab:
                    subwords.append(subword)
                    start = end
                    match = True
                    break
            if not match:  # No subword match found
                subwords.append(self.unk_token)
                break
        return subwords
    
    def decode(self, tokens):
        text = ''
        for token in tokens:
            if token.startswith(self.wordpieces_prefix):
                # Remove the '##' prefix and concatenate without space
                text += token[2:]
            elif token in [self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token]:
                # Skip special tokens if desired, or handle them differently
                continue
            elif token == self.aps_token:
                # Replace [APS] with a ' character
                text += "'"
            elif token == self.space_token:
                # Replace [SPACE] with a space character
                text += ' '
            elif token == self.brk_token:
                # Replace [BRK] with a newline character
                text += '\n'
            else:
                # Add a space before the token if it's not the first token and the last character isn't a newline
                if text and not text.endswith('\n') and not text.endswith("'"):
                    text += ' '
                text += token
        return text
    
    def save(self, path):
       with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'word_freqs': self.word_freqs,
            }, f, ensure_ascii=False)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab =  data['vocab']
            self.vocab_size = len(self.vocab)
            self.word_freqs =  {k: int(v) for k, v in data['word_freqs'].items()}
            
        
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



# TRAIN THE TOKENIZER
# Load the text to train the tokenizer
url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
with urllib.request.urlopen(url) as response:
    # Read the response content
    data = response.read()

    # Decode the bytes to string using utf-8 encoding
    text = data.decode('utf-8')

start = text.find('Chapter I.]')
end = text.rfind('END OF THE PROJECT GUTENBERG EBOOK')
text = text[start:end]

filename = "tokenizer/dataset/merged_clean.txt"
dataset = load_separate_and_clean_stories(filename)

# Train the tokenizer
tokenizer = WordPieceTokenizer(2000)
tokenizer.fit(dataset[0])

# Save the tokenizer
tokenizer.save('tokenizer/wordPieceVocab.json')


# USE THE TOKENIZER
# Load the dataset
filename = "tokenizer/dataset/merged_clean.txt"
# filename = "tokenizer/dataset/combined_stories.txt"
dataset = load_separate_and_clean_stories(filename)

# Load the tokenizer
tokenizer = WordPieceTokenizer()
tokenizer.load('tokenizer/wordPieceVocab.json')

# Encode the first story
story = dataset[0]
#print(story)
# tokens = tokenizer.encode_text(story)
tokens = tokenizer.encode(story)
# print(tokens)


# Decode the tokens
decoded_story = tokenizer.decode(tokens)
print(decoded_story)
