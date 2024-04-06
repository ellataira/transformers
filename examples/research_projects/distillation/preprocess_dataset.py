import datasets
import re
import random

dataset = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True, cache_dir="../", download_config=datasets.DownloadConfig(cache_dir="../"))

# Create a text file to store the compiled text
text_file = open("/scratch/taira.e/c4_10_dataset_distill.txt", "w", encoding="utf-8")
sentence_pattern = re.compile(r'[.!?]|[\n]')

"""
Define the size of the subset you want to create
c4 en train set = 364868892 = 305 gb
"""

subset_size = 1000000 

# Iterate over the dataset and write the text to the file
for i, example in enumerate(dataset):
    if i > subset_size :
        break

    # sentences = [sentence.strip() for sentence in re.split(sentence_pattern, example['text']) if sentence.strip()]
    #
    # # Write each sentence to the file, with each sentence duplicated and separated by a tab
    # for sentence in sentences:
    #     text_file.write(f"{sentence}\n")
    for s in re.split(sentence_pattern, example['text']):
        if s.strip():
            trimmed_sentence = " ".join(s.strip().split()[:512])
            text_file.write(f"{trimmed_sentence}\n")

# Close the file
text_file.close()
print("Wrote dataset of size: ", subset_size)




