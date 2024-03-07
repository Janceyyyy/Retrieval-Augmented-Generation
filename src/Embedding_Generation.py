import json
import torch
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#get embedding for whole training dataset!

# load training dataset
with open('../data/training_tags.json', 'r') as file:
    data = json.load(file)

# extract 'input' 
inputs = [item['input'] for item in data]
# filenames = [item['filename'] for item in data]

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def roberta_embeddings(texts, layer_num=10):
    """Generate embeddings for texts using RoBERTa."""

    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.hidden_states
    target_layer = hidden_states[layer_num]
    mean_embedding = torch.mean(target_layer, dim=1)
    return mean_embedding[0].cpu().numpy()

embeddings = [roberta_embeddings(text) for text in inputs]

np.save('../data/embeddings_bert.npy', embeddings)

