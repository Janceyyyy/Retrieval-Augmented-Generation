import json
import torch
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open('../data/test.json', 'r') as file:
    test = json.load(file)

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name, output_hidden_states=True)
nnum = len(test)#50 unmber of files to run through

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def roberta_embeddings(texts, filename, layer_num=10):
    """Generate embeddings for texts using RoBERTa."""
    
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**encoded_input) 
        hidden_states = outputs.hidden_states
    target_layer = hidden_states[layer_num]
    mean_embedding = torch.mean(target_layer, dim=1)
    return [filename, mean_embedding[0].cpu().numpy()]
with open('training_tags.json', 'r') as file:
    original_data = json.load(file)


saved_embeddings = np.load('../data/embeddings_bert.npy')
test_embeddings = [roberta_embeddings(item['input'], item['filename']) for item in test[:nnum]]

top_n = 1 #number of similar files produced
enhanced_training_data_per_test = []
data_to_save = []

for index, (filename, test_embedding) in enumerate(test_embeddings):
    similarities = cosine_similarity([test_embedding], saved_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # find simmliar samples
    similar_samples = [original_data[i] for i in top_indices]
    #data_to_save.append({'filename': filename, 'input': test[index]['input']})
    save = []

    enhanced_training_data = []
    for sample in similar_samples:
        enhanced_training_data.append({
            'input': sample['input'],
            'output': sample['output']
        })


    print(index, enhanced_training_data)
    enhanced_training_data_per_test.append(enhanced_training_data)
    data_to_save.append([{'filename': filename, 'input': test[index]['input']},save])

with open('similar.json', 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)
