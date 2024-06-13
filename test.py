from lavis.models import load_model_and_preprocess
import torch
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

raw_image = Image.open("/home/wuyinjun/concept_based_retrieval/full_image.png").convert("RGB")
caption = "a large fountain spewing water into the air"
caption2 = "Hello"
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
text_input = txt_processors["eval"](caption)
text_input2 = txt_processors["eval"](caption2)
sample = {"image": image, "text_input": [text_input]}
sample2 = {"image": image, "text_input": [text_input2]}

features_multimodal = model.extract_features(sample)
print(features_multimodal.multimodal_embeds.shape)
# torch.Size([1, 12, 768]), use features_multimodal[:,0,:] for multimodal classification tasks

features_image = model.extract_features(sample, mode="image")
features_image2 = model.extract_features(sample2, mode="image")
features_text = model.extract_features(sample, mode="text")

print(torch.norm(features_image.image_embeds_proj[:,0,:] - features_image2.image_embeds_proj[:,0,:]))

print(features_image.image_embeds.shape)
# torch.Size([1, 197, 768])
print(features_text.text_embeds.shape)
# torch.Size([1, 12, 768])

# low-dimensional projected features
print(features_image.image_embeds_proj.shape)
# torch.Size([1, 197, 256])
print(features_text.text_embeds_proj.shape)
# torch.Size([1, 12, 256])
similarity = features_image.image_embeds_proj[:,0,:] @ features_text.text_embeds_proj[:,0,:].t()
print(similarity)
# tensor([[0.2622]])