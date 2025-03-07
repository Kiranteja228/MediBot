from transformers import pipeline

pipe = pipeline("image-classification", model="Heem2/bone-fracture-detection-using-xray")
preds = pipe(images="D:\.VIT_self\Medical_AI_agent\images\image2.png")
print(preds)