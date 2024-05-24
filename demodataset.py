from datasets import load_dataset

#
# download dataset BUAADreamer/mllm_demo from huggingface
# https://huggingface.co/datasets/BUAADreamer/mllm_demo
# C:\Users\changzheng\.cache\huggingface\datasets\BUAADreamer___mllm_demo
# 
dataset = load_dataset("BUAADreamer/mllm_demo")

#
# dataset.info
print("**** dataset:")
print(dataset)
print("**** dataset.keys():")
print(dataset.keys())

#
train_dataset = dataset['train']

#
for sample in train_dataset:
    print(sample['messages'])
    print(sample['images'])


import os

# 创建保存图像的目录
os.makedirs('images', exist_ok=True)

# 循环遍历样本，保存图像
for i, sample in enumerate(train_dataset):
    images = sample['images']
    for j, image in enumerate(images):
        image_path = f'images/sample_{i}_image_{j}.jpg'  # 图像保存路径
        image.save(image_path)  # 保存图像到指定路径
        print(f'Saved image {image_path}')