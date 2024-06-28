from dessert_minheap import DocRetrieval
import torch
# hashes_per_table: int, num_tables: int, dense_input_dimension: int, centroids: np.ndarray
hashes_per_table=2
num_tables=2
dense_input_dimension=100
centroids = torch.zeros([1,dense_input_dimension])
retrieval_method = DocRetrieval(2, 2, 100, centroids)

doc_embds_ls = []

for idx in range(10):
    doc_embds_ls.append(torch.rand(20,dense_input_dimension))

for idx in range(len(doc_embds_ls)):
  retrieval_method.add_doc(doc_embds_ls[idx].numpy(), idx)
  
result = retrieval_method.query(torch.rand(15,dense_input_dimension).numpy(), top_k=5, num_to_rerank=5)
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
