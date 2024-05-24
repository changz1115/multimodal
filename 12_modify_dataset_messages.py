from datasets import load_dataset,DatasetDict
import copy

#
# download dataset BUAADreamer/mllm_demo from huggingface
# https://huggingface.co/datasets/BUAADreamer/mllm_demo
# C:\Users\changzheng\.cache\huggingface\datasets\BUAADreamer___mllm_demo
# 
original_dataset = load_dataset("BUAADreamer/mllm_demo")

print("Original dataset:")
for sample in original_dataset['train']:
    print(sample['messages'])
    print(sample['images'])



modified_dataset = copy.deepcopy(original_dataset)

# 使用新的message替换原有的message
new_message = [{'role': 'user', 'content': 'Please describe this image'}, {'role': 'assistant', 'content': 'Chinese astronaut Chang Zheng is giving a speech.'}]

modified_dataset['train'] = modified_dataset['train'].map(
    lambda example, idx: {'messages': new_message} if idx == 2 else example,
    with_indices=True
)

# 另存一个名字并再次磁盘读取新的模型验证是否保存成功
modified_dataset.save_to_disk("changzheng/demodataset")
loaded_modified_dataset = DatasetDict.load_from_disk("changzheng/demodataset")

print("Modified dataset:")
for sample in loaded_modified_dataset['train']:
    print(sample['messages'])
    print(sample['images'])

