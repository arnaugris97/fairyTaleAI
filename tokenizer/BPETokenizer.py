from collections import defaultdict, Counter
import json
import re

import requests


class BytePairEncodingTokenizer():
    def __init__(self, num_merges=10000):
        self.num_merges = num_merges
        self.vocab = {}
        self.word2idx = {}
        self.idx2word = {}

    def fit(self, text):
         # Initialize vocabulary with word frequency
        words = re.findall(r'\w+|[^\w\s]', text)
        word_freq = Counter(words)
        vocab = defaultdict(int)
        for word, freq in word_freq.items():
            chars = ' '.join(word)  # Split word into characters
            chars += ' </w>'  # Add end word token
            vocab[chars] += freq

        # Main loop to perform merges
        for _ in range(self.num_merges):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[(symbols[i], symbols[i + 1])] += freq

            if not pairs:
                break

            # Find the most frequent pair
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                continue

            # Merge the most frequent pair
            new_symbol = ''.join(best_pair)
            new_vocab = defaultdict(int)
            for word in vocab:
                new_word = word.replace(' '.join(best_pair), new_symbol)
                new_vocab[new_word] += vocab[word]
            vocab = new_vocab

        # Create a set of unique tokens
        tokens = set()
        for word in vocab.keys():
            for symbol in word.split():
                tokens.add(symbol)

        # Assign indexes to tokens
        self.vocab = {token: idx for idx, token in enumerate(tokens)}
        self.word2idx = {word: idx for idx, word in enumerate(tokens)}
        self.idx2word = {idx: word for idx, word in enumerate(tokens)}

      
        

    def encode(self, text):
        words = re.findall(r'\w+|[^\w\s]', text)

        tokens = []
        for word in words:
            # Append the end of word token
            word += '</w>'

            start = 0
            subtokens = []
            while start < len(word):
                longest_match = None
                # Check for the longest sequence in the vocabulary starting from 'start'
                for end in range(start + 1, len(word) + 1):
                    substring = word[start:end]
                    if substring in self.word2idx:
                        longest_match = substring

                if longest_match is not None:
                    subtokens.append(longest_match)
                    start += len(longest_match)
                else:
                    # If no match found, increment to try the next character
                    start += 1
        
            # Translate subtokens into indices
            tokens.extend(self.word2idx[token] for token in subtokens if token in self.word2idx)

        return tokens

    def decode(self, indices):
        # Translate indices back to tokens using the idx2word mapping
        tokens = [self.idx2word[index] for index in indices]

        # Reconstruct the original text from tokens
        text = ''
        for token in tokens:
            if token.endswith('</w>'):
                # Remove the end-of-word marker and ensure space only if it's not punctuation
                if token[:-4] == '' or token[:-4][-1].isspace():
                    text += token[:-4]  # Add nothing more, just the punctuation or space
                else:
                    text += token[:-4] + ' '  # Add the word and a space
            else:
                text += token  # Add the token as it is (for parts of words or ongoing words without spaces)

        return text.strip()  # Trim any excess whitespace from the ends

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return len(self.vocab)

    def get_vocab_list(self):
        return list(self.vocab.keys())

    def save(self, path):
       with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'num_merges': self.num_merges
            }, f, ensure_ascii=False)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab =  {k: int(v) for k, v in data['vocab'].items()}
            self.word2idx =  {k: int(v) for k, v in data['word2idx'].items()}
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            self.num_merges = data['num_merges']

    def __len__(self):
        return len(self.vocab)
    


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
# Download "Pride and Prejudice" from Project Gutenberg
url = 'https://www.gutenberg.org/files/1342/1342-0.txt'
response = requests.get(url)
text = response.text

start = text.find('Chapter I.]')
end = text.rfind('END OF THE PROJECT GUTENBERG EBOOK')
text = text[start:end]

# Train the tokenizer
tokenizer = BytePairEncodingTokenizer(num_merges=100000)
tokenizer.fit(text)

# # Save the tokenizer
tokenizer.save("bpe_tokenizer.json")


# Load the tokenizer
loaded_tokenizer = BytePairEncodingTokenizer()
loaded_tokenizer.load("bpe_tokenizer.json")

# Check if the vocabularies are the same
assert tokenizer.get_vocab() == loaded_tokenizer.get_vocab()
assert tokenizer.get_vocab_size() == loaded_tokenizer.get_vocab_size()
assert tokenizer.get_vocab_list() == loaded_tokenizer.get_vocab_list()

print("Vocabularies are the same!")



# LOAD AND USE THE TOKENIZER
# Load the dataset
filename = "dataset/merged_clean.txt"
dataset = load_separate_and_clean_stories(filename)

# Load the tokenizer
tokenizer = BytePairEncodingTokenizer()
tokenizer.load("bpe_tokenizer.json")

# Encode a text
encoded = tokenizer.encode(dataset[0][:10000])

print(f'original word:', dataset[0][:10000])
print(f'encoded:', encoded)

decoded = tokenizer.decode(encoded)
print(f'decoded:', decoded)




