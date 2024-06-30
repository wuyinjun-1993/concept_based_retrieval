from dessert_minheap_torch import DocRetrieval
import torch
import random
import numpy as np
import os

def set_rand_seed(seed_value):
    # Set seed for Python's built-in random module
    random.seed(seed_value)
    
    # Set seed for NumPy
    np.random.seed(seed_value)
    
    # Set seed for PyTorch
    torch.manual_seed(seed_value)
    
    # Set seed for CUDA (if using a GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # If using multi-GPU.
    
    # Ensure deterministic operations for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# hashes_per_table: int, num_tables: int, dense_input_dimension: int, centroids: np.ndarray
set_rand_seed(100)
hashes_per_table=2
num_tables=100
dense_input_dimension=100
centroids = torch.zeros([1,dense_input_dimension])
doc_num=10

if os.path.exists('output/doc_embds_ls.pt'):
  doc_embds_ls = torch.load('output/doc_embds_ls.pt')
else:
  doc_embds_ls = []
  for idx in range(doc_num):
      data = torch.rand(20,dense_input_dimension)
      print(data)
      doc_embds_ls.append(data)
  torch.save(doc_embds_ls, 'output/doc_embds_ls.pt')

print("document count::", len(doc_embds_ls))


full_doc_embeds = torch.eye(100)
doc_embds_ls = [full_doc_embeds[i*10:(i+1)*10] for i in range(10)]

centroids = [torch.mean(torch.cat(doc_embds_ls[i:i+2]), dim=0) for i in range(0, len(doc_embds_ls), 2)]

retrieval_method = DocRetrieval(doc_num, hashes_per_table, num_tables, 100, torch.stack(centroids))

for idx in range(len(doc_embds_ls)):
  retrieval_method.add_doc(doc_embds_ls[idx], idx)

print("similarity::", torch.nn.functional.cosine_similarity(doc_embds_ls[-1].unsqueeze(1), doc_embds_ls[-2].unsqueeze(0)).max())
# print("similarity::", torch.nn.functional.cosine_similarity(doc_embds_ls[-1], doc_embds_ls[-2]))
# if os.path.exists('output/query_data.pt'):
#   query_data = torch.load('output/query_data.pt')
# else:
#   query_data  = torch.rand(15,dense_input_dimension)
#   torch.save(query_data, 'output/query_data.pt')
# print(query_data)
# _,result = retrieval_method.query(doc_embds_ls[-1], top_k=5, num_to_rerank=5)
result = retrieval_method.query_multi_queries([[doc_embds_ls[-1]]], top_k=5, num_to_rerank=5)
# result = retrieval_method.query(doc_embds_ls[-1].numpy(), top_k=5, num_to_rerank=len(doc_embds_ls))

print(result)

# from openai import OpenAI
# client = OpenAI()




# response = client.images.generate(
#   model="dall-e-3",
#   prompt="taxi parking on a road in with city buildings around and some people on the side walk",
#   size="1024x1024",
#   quality="standard",
#   style="natural",
#   n=1,
# )

# image_url = response.data[0].url
# print(image_url)

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Specify the model name (you can choose any model compatible with AutoModelForCausalLM)
# model_name = "princeton-nlp/Sheared-LLaMA-1.3B"

# # Load the pre-trained model and tokenizer
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Set the model to evaluation mode
# model.eval()

# # Define the input prompt
# prompt = "Once upon a time, in a land far, far away"

# # Tokenize the input prompt
# input_ids = tokenizer.encode(prompt, return_tensors="pt")

# # Generate text
# output_ids = model.generate(input_ids, max_length=100, num_return_sequences=1)

# # Decode the generated text
# generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # Print the generated text
# print(generated_text)
