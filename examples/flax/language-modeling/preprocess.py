from transformers import RobertaTokenizer, RobertaConfig

tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
config = RobertaConfig.from_pretrained("FacebookAI/roberta-base")

tokenizer.save_pretrained("scratch/taira.e/english-roberta-base")
config.save_pretrained("scratch/taira.e/english-roberta-base")


dataset_file = "/scratch/taira.e/c4_10_dataset_distill.txt"
with open(dataset_file, "r", encoding="utf-8") as f:
    sentences = f.readlines()

# Tokenize the sentences and create masked language modeling examples
processed_examples = []
for sentence in sentences:
    # Tokenize the sentence
    tokens = tokenizer.tokenize(sentence.strip())

    # Create masked language modeling examples
    for i, token in enumerate(tokens):
        # Randomly mask some tokens
        if i % 5 == 0:  # Adjust the masking probability as needed
            masked_token = "[MASK]"
            masked_index = i
            tokens[masked_index] = masked_token

    # Convert tokens back to string
    processed_sentence = tokenizer.convert_tokens_to_string(tokens)

    # Add the processed example to the list
    processed_examples.append(processed_sentence)

# Save the processed examples to a new text file
output_file = "/scratch/taira.e/english-roberta-base/processed_roberta_ds.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(processed_examples))
