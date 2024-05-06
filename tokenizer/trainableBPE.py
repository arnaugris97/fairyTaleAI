import re
from tokenizers import decoders, models, pre_tokenizers, processors, trainers, Tokenizer


def separate_and_clean_stories(filename):
    with open(filename, 'r') as file:
        content = file.read()

    stories = content.split('\n\n\n\n')

    cleaned_stories = []
    for story in stories:
        cleaned_story = re.sub(r'\n\s*\n', '\n', story.strip())
        cleaned_stories.append(cleaned_story)

    num_stories = len(cleaned_stories)
    
    return num_stories, cleaned_stories



filename = "dataset/merged_clean.txt"
num_stories, stories = separate_and_clean_stories(filename)
print(f"Number of stories: {num_stories}")


batch_size = 1000
all_texts = [stories[i : i + batch_size] for i in range(0, len(stories), batch_size)]


def batch_iterator():
    for i in range(0, len(stories), batch_size):
        yield stories[i : i + batch_size]

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)



trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()


# Encode a text
encoded = tokenizer.encode(stories[0])

# Print the encoded tokens and ids
print("Tokens: ", encoded.tokens)
print("IDs: ", encoded.ids)

# Decode the text
decoded = tokenizer.decode(encoded.ids)

# Print the decoded text
print("Decoded: ", decoded)