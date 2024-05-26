import torch # type: ignore
import random

# Helper function to prepare tensors
def prepare_tensor(data, device):
    """Convert data to tensor, add batch dimension, and move to specified device."""
    return torch.tensor(data).unsqueeze(0).to(device)

# Function to create sentence pairs with NSP labels
def create_sentence_pair(sentences, index):
    """Create sentence pairs with NSP labels.
    
    Args:
        sentences (list of str): List of sentences.
        index (int): Index of the first sentence in the pair.

    Returns:
        tuple: (sentence1, sentence2, nsp_label)
               sentence1 (str): The first sentence.
               sentence2 (str): The second sentence, which could be a random sentence.
               nsp_label (int): NSP label, 1 if sentence2 follows sentence1, 0 otherwise.
    """
    # Decide whether to create a positive pair (50% chance)
    if random.random() > 0.5:
        # Positive pair: sentence B follows sentence A
        sentence1 = sentences[index]
        sentence2 = sentences[index + 1]
        nsp_label = 1  # Label for a positive pair
    else:
        # Negative pair: sentence B is a random sentence
        sentence1 = sentences[index]
        random_index = random.randint(0, len(sentences) - 1)
        # Ensure the random sentence is not the same as the current or next sentence
        while random_index == index or random_index == index + 1:
            random_index = random.randint(0, len(sentences) - 1)
        sentence2 = sentences[random_index]
        nsp_label = 0  # Label for a negative pair

    return sentence1, sentence2, nsp_label
