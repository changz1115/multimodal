```log
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
06/01/2024 14:48:07 - INFO - llamafactory.hparams.parser - Process rank: 0, device: cuda:0, n_gpu: 1, distributed training: False, compute dtype: torch.float16
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:08,551 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer.model
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:08,552 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:08,552 >> loading file added_tokens.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/added_tokens.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:08,552 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/special_tokens_map.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:08,552 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer_config.json
[WARNING|logging.py:314] 2024-06-01 14:48:08,652 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|image_processing_utils.py:374] 2024-06-01 14:48:10,011 >> loading configuration file preprocessor_config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/preprocessor_config.json
[INFO|image_processing_utils.py:374] 2024-06-01 14:48:10,417 >> loading configuration file preprocessor_config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/preprocessor_config.json
[INFO|image_processing_utils.py:424] 2024-06-01 14:48:10,420 >> Image processor CLIPImageProcessor {
  "_valid_processor_keys": [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_center_crop",
    "crop_size",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
    "return_tensors",
    "data_format",
    "input_data_format"
  ],
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "processor_class": "LlavaProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 336
  }
}

[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:10,828 >> loading file tokenizer.model from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer.model
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:10,828 >> loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:10,828 >> loading file added_tokens.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/added_tokens.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:10,828 >> loading file special_tokens_map.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/special_tokens_map.json
[INFO|tokenization_utils_base.py:2108] 2024-06-01 14:48:10,828 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/tokenizer_config.json
[WARNING|logging.py:314] 2024-06-01 14:48:10,894 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
[INFO|processing_utils.py:400] 2024-06-01 14:48:11,309 >> Processor LlavaProcessor:
- image_processor: CLIPImageProcessor {
  "_valid_processor_keys": [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_center_crop",
    "crop_size",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
    "return_tensors",
    "data_format",
    "input_data_format"
  ],
  "crop_size": {
    "height": 336,
    "width": 336
  },
  "do_center_crop": true,
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "CLIPImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "processor_class": "LlavaProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 336
  }
}

- tokenizer: LlamaTokenizerFast(name_or_path='llava-hf/llava-1.5-7b-hf', vocab_size=32000, model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='left', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
	0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32000: AddedToken("<image>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
	32001: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}

{
  "processor_class": "LlavaProcessor"
}

06/01/2024 14:48:11 - INFO - llamafactory.data.loader - Loading dataset mllm_demo.json...
Generating train split: 6 examples [00:00, 158.43 examples/s]
num_proc must be <= 6. Reducing num_proc to 6 for dataset of size 6.
Converting format of dataset (num_proc=6): 100%|█| 6/6 [00:00<00:00, 50.87 examp
num_proc must be <= 6. Reducing num_proc to 6 for dataset of size 6.
Running tokenizer on dataset (num_proc=6): 100%|█| 6/6 [00:00<00:00, 17.43 examp
input_ids:
[319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871, 32000, 11644, 526, 896, 29973, 319, 1799, 9047, 13566, 29901, 2688, 29915, 276, 476, 1662, 322, 402, 2267, 29920, 1335, 515, 19584, 13564, 436, 29889, 2, 3148, 1001, 29901, 1724, 526, 896, 2599, 29973, 319, 1799, 9047, 13566, 29901, 2688, 526, 10894, 1218, 373, 278, 269, 11953, 1746, 29889, 2]
inputs:
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image> Who are they? ASSISTANT: They're Kane and Gretzka from Bayern Munich.</s> USER: What are they doing? ASSISTANT: They are celebrating on the soccer field.</s>
label_ids:
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2688, 29915, 276, 476, 1662, 322, 402, 2267, 29920, 1335, 515, 19584, 13564, 436, 29889, 2, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2688, 526, 10894, 1218, 373, 278, 269, 11953, 1746, 29889, 2]
labels:
They're Kane and Gretzka from Bayern Munich.</s> They are celebrating on the soccer field.</s>
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
[INFO|configuration_utils.py:733] 2024-06-01 14:48:16,148 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/config.json
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py:100: FutureWarning: The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect
  warnings.warn(
[INFO|configuration_utils.py:796] 2024-06-01 14:48:16,158 >> Model config LlavaConfig {
  "_name_or_path": "llava-hf/llava-1.5-7b-hf",
  "architectures": [
    "LlavaForConditionalGeneration"
  ],
  "ignore_index": -100,
  "image_token_index": 32000,
  "model_type": "llava",
  "pad_token_id": 32001,
  "projector_hidden_act": "gelu",
  "text_config": {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "rms_norm_eps": 1e-05,
    "torch_dtype": "float16",
    "vocab_size": 32064
  },
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.1",
  "vision_config": {
    "hidden_size": 1024,
    "image_size": 336,
    "intermediate_size": 4096,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "vocab_size": 32000
  },
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "default"
}

[INFO|modeling_utils.py:3474] 2024-06-01 14:48:16,166 >> loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/model.safetensors.index.json
[INFO|modeling_utils.py:1519] 2024-06-01 14:48:16,168 >> Instantiating LlavaForConditionalGeneration model under default dtype torch.float16.
[INFO|configuration_utils.py:962] 2024-06-01 14:48:16,169 >> Generate config GenerationConfig {
  "pad_token_id": 32001
}

[INFO|configuration_utils.py:962] 2024-06-01 14:48:16,310 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}

Loading checkpoint shards: 100%|██████████████████| 3/3 [00:33<00:00, 11.29s/it]
[INFO|modeling_utils.py:4280] 2024-06-01 14:48:50,288 >> All model checkpoint weights were used when initializing LlavaForConditionalGeneration.

[INFO|modeling_utils.py:4288] 2024-06-01 14:48:50,288 >> All the weights of LlavaForConditionalGeneration were initialized from the model checkpoint at llava-hf/llava-1.5-7b-hf.
If your task is similar to the task the model of the checkpoint was trained on, you can already use LlavaForConditionalGeneration for predictions without further training.
[INFO|configuration_utils.py:917] 2024-06-01 14:48:50,765 >> loading configuration file generation_config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/generation_config.json
[INFO|configuration_utils.py:962] 2024-06-01 14:48:50,766 >> Generate config GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2,
  "pad_token_id": 32001
}

06/01/2024 14:48:50 - INFO - llamafactory.model.utils.checkpointing - Gradient checkpointing enabled.
06/01/2024 14:48:50 - INFO - llamafactory.model.utils.attention - Using torch SDPA for faster training and inference.
06/01/2024 14:48:50 - INFO - llamafactory.model.adapter - Upcasting trainable params to float32.
06/01/2024 14:48:50 - INFO - llamafactory.model.adapter - Fine-tuning method: LoRA
06/01/2024 14:48:52 - INFO - llamafactory.model.loader - trainable params: 4194304 || all params: 7067621376 || trainable%: 0.0593
[INFO|trainer.py:641] 2024-06-01 14:48:52,271 >> Using auto half precision backend
[INFO|trainer.py:2078] 2024-06-01 14:48:52,391 >> ***** Running training *****
[INFO|trainer.py:2079] 2024-06-01 14:48:52,391 >>   Num examples = 5
[INFO|trainer.py:2080] 2024-06-01 14:48:52,391 >>   Num Epochs = 3
[INFO|trainer.py:2081] 2024-06-01 14:48:52,391 >>   Instantaneous batch size per device = 1
[INFO|trainer.py:2084] 2024-06-01 14:48:52,391 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
[INFO|trainer.py:2085] 2024-06-01 14:48:52,391 >>   Gradient Accumulation steps = 8
[INFO|trainer.py:2086] 2024-06-01 14:48:52,391 >>   Total optimization steps = 3
[INFO|trainer.py:2087] 2024-06-01 14:48:52,393 >>   Number of trainable parameters = 4,194,304
  0%|                                                     | 0/3 [00:00<?, ?it/s]/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/torch/utils/checkpoint.py:61: UserWarning: None of the inputs have requires_grad=True. Gradients will be None
  warnings.warn(
[WARNING|logging.py:329] 2024-06-01 14:48:53,723 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
100%|█████████████████████████████████████████████| 3/3 [00:12<00:00,  3.61s/it][INFO|trainer.py:2329] 2024-06-01 14:49:04,894 >> 

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': 12.5009, 'train_samples_per_second': 1.2, 'train_steps_per_second': 0.24, 'train_loss': 0.7740340232849121, 'epoch': 2.0}
100%|█████████████████████████████████████████████| 3/3 [00:12<00:00,  4.17s/it]
[INFO|trainer.py:3410] 2024-06-01 14:49:04,895 >> Saving model checkpoint to saves/llava1_5-7b/lora/sft
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/llava/configuration_llava.py:140: FutureWarning: The `vocab_size` attribute is deprecated and will be removed in v4.42, Please use `text_config.vocab_size` instead.
  warnings.warn(
[INFO|configuration_utils.py:733] 2024-06-01 14:49:05,801 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots/05ae2434cbb430be33edcba0c5203e7023f785b7/config.json
[INFO|configuration_utils.py:796] 2024-06-01 14:49:05,805 >> Model config LlavaConfig {
  "architectures": [
    "LlavaForConditionalGeneration"
  ],
  "ignore_index": -100,
  "image_token_index": 32000,
  "model_type": "llava",
  "pad_token_id": 32001,
  "projector_hidden_act": "gelu",
  "text_config": {
    "_name_or_path": "lmsys/vicuna-7b-v1.5",
    "architectures": [
      "LlamaForCausalLM"
    ],
    "max_position_embeddings": 4096,
    "model_type": "llama",
    "rms_norm_eps": 1e-05,
    "torch_dtype": "float16",
    "vocab_size": 32064
  },
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.1",
  "vision_config": {
    "hidden_size": 1024,
    "image_size": 336,
    "intermediate_size": 4096,
    "model_type": "clip_vision_model",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "projection_dim": 768,
    "vocab_size": 32000
  },
  "vision_feature_layer": -2,
  "vision_feature_select_strategy": "default"
}

[INFO|tokenization_utils_base.py:2513] 2024-06-01 14:49:05,830 >> tokenizer config file saved in saves/llava1_5-7b/lora/sft/tokenizer_config.json
[INFO|tokenization_utils_base.py:2522] 2024-06-01 14:49:05,830 >> Special tokens file saved in saves/llava1_5-7b/lora/sft/special_tokens_map.json
[INFO|image_processing_utils.py:257] 2024-06-01 14:49:05,865 >> Image processor saved in saves/llava1_5-7b/lora/sft/preprocessor_config.json
***** train metrics *****
  epoch                    =        2.0
  total_flos               =    47747GF
  train_loss               =      0.774
  train_runtime            = 0:00:12.50
  train_samples_per_second =        1.2
  train_steps_per_second   =       0.24
06/01/2024 14:49:05 - WARNING - llamafactory.extras.ploting - No metric loss to plot.
06/01/2024 14:49:05 - WARNING - llamafactory.extras.ploting - No metric eval_loss to plot.
[INFO|trainer.py:3719] 2024-06-01 14:49:05,870 >> ***** Running Evaluation *****
[INFO|trainer.py:3721] 2024-06-01 14:49:05,870 >>   Num examples = 1
[INFO|trainer.py:3724] 2024-06-01 14:49:05,870 >>   Batch size = 1
100%|███████████████████████████████████████████| 1/1 [00:00<00:00, 3572.66it/s]
***** eval metrics *****
  epoch                   =        2.0
  eval_loss               =     2.6885
  eval_runtime            = 0:00:00.43
  eval_samples_per_second =      2.315
  eval_steps_per_second   =      2.315
[INFO|modelcard.py:450] 2024-06-01 14:49:06,303 >> Dropping the following result as it does not have all the necessary fields:
{'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}
```