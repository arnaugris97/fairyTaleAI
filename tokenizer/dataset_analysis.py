from BPETokenizer import load_separate_and_clean_stories
from collections import defaultdict, Counter
import json
import re

import requests

filename = "dataset/merged_clean.txt"
print(filename)
#dataset = load_separate_and_clean_stories(filename)
