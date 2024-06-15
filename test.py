import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForMaskedLM


torch.random.manual_seed(0)

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct", 
#     device_map="cuda", 
#     torch_dtype="auto", 
#     trust_remote_code=True, 
# )
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")
# model = AutoModel.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForMaskedLM.from_pretrained(model_name)
model0 = DistilBertForSequenceClassification.from_pretrained(model_name)
model0 = model0.to("cuda")


collated_texts = tokenizer(["Can you provide ways to eat combinations of bananas and dragonfruits?", "hello"], 
                     return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True)
                    #  return_tensors="pt")

# collated_texts['input_ids'] = [tokenizer.encode(prefix, add_special_tokens=False) + input_ids +
#                                        self.tokenizer.encode(suffix, add_special_tokens=False)
#                                        for input_ids in collated_texts['input_ids']]

features = tokenizer.pad(
    collated_texts,
    padding=True,
    pad_to_multiple_of=16,
    return_attention_mask=True,
    return_tensors='pt',
)

model = model.to("cuda")

features["input_ids"] = features["input_ids"].to("cuda")
features["attention_mask"] = features["attention_mask"].to("cuda")
output = model(input_ids=features["input_ids"], attention_mask=features["attention_mask"], output_hidden_states=True,
            return_dict=True)

output_feature = output.hidden_states[-1]
print(features)



class CustomDistilBertForMaskedLM(DistilBertForMaskedLM):
    def forward(self, custom_embeddings, attention_mask=None):
        # Pass the custom embeddings through the transformer layers
        # transformer_outputs = self.distilbert.transformer(custom_embeddings, attn_mask=attention_mask, head_mask=attention_mask)
        # hidden_states = transformer_outputs[0]
        
        # Pass through the language modeling head
        prediction_logits = self.vocab_transform(custom_embeddings)
        prediction_logits = self.activation(prediction_logits)
        prediction_logits = self.vocab_layer_norm(prediction_logits)
        prediction_logits = self.vocab_projector(prediction_logits)
        
        return prediction_logits

custom_embeddings = torch.randn(1, 10, 768)

custom_model = CustomDistilBertForMaskedLM.from_pretrained(model_name)
attention_mask = torch.ones(custom_embeddings.shape)

with torch.no_grad():
    logits = custom_model(custom_embeddings, attention_mask=attention_mask)

# from lavis.models import load_model_and_preprocess
# import torch
# from PIL import Image
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# raw_image = Image.open("/home/wuyinjun/concept_based_retrieval/full_image.png").convert("RGB")
# caption = "a large fountain spewing water into the air"
# caption2 = "Hello"
# image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# text_input = txt_processors["eval"](caption)
# text_input2 = txt_processors["eval"](caption2)
# sample = {"image": image, "text_input": [text_input]}
# sample2 = {"image": image, "text_input": [text_input2]}

# features_multimodal = model.extract_features(sample)
# print(features_multimodal.multimodal_embeds.shape)
# # torch.Size([1, 12, 768]), use features_multimodal[:,0,:] for multimodal classification tasks

# features_image = model.extract_features(sample, mode="image")
# features_image2 = model.extract_features(sample2, mode="image")
# features_text = model.extract_features(sample, mode="text")

# print(torch.norm(features_image.image_embeds_proj[:,0,:] - features_image2.image_embeds_proj[:,0,:]))

# print(features_image.image_embeds.shape)
# # torch.Size([1, 197, 768])
# print(features_text.text_embeds.shape)
# # torch.Size([1, 12, 768])

# # low-dimensional projected features
# print(features_image.image_embeds_proj.shape)
# # torch.Size([1, 197, 256])
# print(features_text.text_embeds_proj.shape)
# # torch.Size([1, 12, 256])
# similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
# print(similarity)
# # tensor([[0.2622]])