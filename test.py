from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="Draw two desks in the same figure that are far away from each other, one has a keyboard and a computer monitor on it while the other one has a printer and fax machine on it. Besides, there is a chair against the wall",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
print(image_url)

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
