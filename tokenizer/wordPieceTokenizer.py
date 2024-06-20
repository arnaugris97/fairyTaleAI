import json
import random
import re
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
        self.word2idx = {}
        self.idx2word = {}

   
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
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        
    
    
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
    
        
        # Convert tokens to ids
        token_ids = []
        token_ids.extend(self.word2idx[token] for token in tokens if token in self.word2idx)

        return token_ids

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
    
    def add_special_tokens(self, token_ids1, token_ids2, max_length=60):
        tokens_with_special_tokens  = [self.word2idx[self.cls_token]] + token_ids1 + [self.word2idx[self.sep_token]] + token_ids2 + [self.word2idx[self.sep_token]]

        # Create attention mask
        attention_mask = [0] * len(tokens_with_special_tokens)

        # Create token segment type ids
        token_type_ids = [1] * (len(token_ids1) + 2) + [2] * (len(token_ids2) + 1)
        
        padded_token_ids = tokens_with_special_tokens + [self.word2idx[self.pad_token]] * (max_length - len(tokens_with_special_tokens))
        attention_mask = attention_mask + [1] * (max_length - len(attention_mask))
        token_type_ids = token_type_ids + [0] * (max_length - len(token_type_ids))
        
        return padded_token_ids, attention_mask, token_type_ids
    
        tokens = [self.idx2word[index] for index in indices]
        # Split the text by the first sep_token
        sep_index = tokens.index(self.sep_token)
        sentence1 = tokens[1:sep_index]
        sentence2 = tokens[sep_index + 1:]
        text1 = ''
        text2 = ''
        # Perform for loop for both sentences at the same time
        

        for token in sentence1:
            if token.startswith(self.wordpieces_prefix):
                # Remove the '##' prefix and concatenate without space
                text1 += token[2:]
            elif token in [self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token]:
                # Skip special tokens if desired, or handle them differently
                continue
            elif token == self.aps_token:
                # Replace [APS] with a ' character
                text1 += "'"
            elif token == self.space_token:
                # Replace [SPACE] with a space character
                text1 += ' '
            elif token == self.brk_token:
                # Replace [BRK] with a newline character
                text1 += '\n'
            else:
                # Add a space before the token if it's not the first token and the last character isn't a newline
                if text1 and not text1.endswith('\n') and not text1.endswith("'"):
                    text1 += ' '
                text1 += token

        for token in sentence1:
            if token.startswith(self.wordpieces_prefix):
                # Remove the '##' prefix and concatenate without space
                text2 += token[2:]
            elif token in [self.unk_token, self.cls_token, self.sep_token, self.pad_token, self.mask_token]:
                # Skip special tokens if desired, or handle them differently
                continue
            elif token == self.aps_token:
                # Replace [APS] with a ' character
                text2 += "'"
            elif token == self.space_token:
                # Replace [SPACE] with a space character
                text2 += ' '
            elif token == self.brk_token:
                # Replace [BRK] with a newline character
                text2 += '\n'
            else:
                # Add a space before the token if it's not the first token and the last character isn't a newline
                if text2 and not text2.endswith('\n') and not text2.endswith("'"):
                    text2 += ' '
                text2 += token

        return text1, text2
    
    def save(self, path):
       with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'word_freqs': self.word_freqs,
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
            }, f, ensure_ascii=False)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab =  data['vocab']
            self.vocab_size = len(self.vocab)
            self.word_freqs =  {k: int(v) for k, v in data['word_freqs'].items()}
            self.word2idx =  {k: int(v) for k, v in data['word2idx'].items()}
            self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
            
        
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

def mask_tokens(token_ids, tokenizer):
    gt = token_ids.copy()

    # Mask 15% of the tokens
    masked_indices = set()
    # 15% of significant tokens, different to [CLS], [SEP], and [PAD]
    significant_tokens = [token for token in token_ids if token not in [tokenizer.word2idx[tokenizer.cls_token], tokenizer.word2idx[tokenizer.sep_token], tokenizer.word2idx[tokenizer.pad_token]]]
    
    num_masked = max(1, int(len(significant_tokens) * 0.15))
    while len(masked_indices) < num_masked:
        index = random.randint(1, len(token_ids) - 2)
        if index in masked_indices:
            continue
        token = token_ids[index]
        if token in [tokenizer.word2idx[tokenizer.cls_token], tokenizer.word2idx[tokenizer.sep_token], tokenizer.word2idx[tokenizer.pad_token]]:
            continue
        token_ids[index] = tokenizer.word2idx[tokenizer.mask_token]
        masked_indices.add(index)
    
    labels = [0 if i not in masked_indices else gt[i] for i in range(len(token_ids))]
 
    return token_ids, labels


def load_separate_and_clean_stories(filename):
    with open(filename, 'r') as file:
        content = file.read()

    stories = content.split('\n\n\n\n')

    cleaned_stories = []
    for story in stories:
        cleaned_story = re.sub(r'\n\s*\n', '\n', story.strip())
        cleaned_stories.append(cleaned_story)
    
    return cleaned_stories