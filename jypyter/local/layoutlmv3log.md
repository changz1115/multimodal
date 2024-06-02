```log
/root/LLaMA-Factory
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
06/02/2024 00:04:22 - INFO - llamafactory.hparams.parser - Process rank: 0, device: cuda:0, n_gpu: 1, distributed training: False, compute dtype: torch.float16
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file vocab.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/vocab.json
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file merges.txt from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/merges.txt
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file tokenizer.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:23,156 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/tokenizer_config.json
[INFO|image_processing_utils.py:374] 2024-06-02 00:04:24,558 >> loading configuration file preprocessor_config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/preprocessor_config.json
[INFO|feature_extraction_utils.py:538] 2024-06-02 00:04:24,933 >> loading configuration file preprocessor_config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/preprocessor_config.json
/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
[INFO|configuration_utils.py:733] 2024-06-02 00:04:25,786 >> loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/config.json
[INFO|configuration_utils.py:796] 2024-06-02 00:04:25,789 >> Model config LayoutLMv3Config {
  "_name_or_path": "microsoft/layoutlmv3-base",
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "coordinate_size": 128,
  "eos_token_id": 2,
  "has_relative_attention_bias": true,
  "has_spatial_attention_bias": true,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "input_size": 224,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_2d_position_embeddings": 1024,
  "max_position_embeddings": 514,
  "max_rel_2d_pos": 256,
  "max_rel_pos": 128,
  "model_type": "layoutlmv3",
  "num_attention_heads": 12,
  "num_channels": 3,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "patch_size": 16,
  "rel_2d_pos_bins": 64,
  "rel_pos_bins": 32,
  "second_input_size": 112,
  "shape_size": 128,
  "text_embed": true,
  "torch_dtype": "float32",
  "transformers_version": "4.41.1",
  "type_vocab_size": 1,
  "visual_embed": true,
  "vocab_size": 50265
}

[INFO|image_processing_utils.py:374] 2024-06-02 00:04:26,400 >> loading configuration file preprocessor_config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/preprocessor_config.json
[INFO|image_processing_utils.py:737] 2024-06-02 00:04:26,400 >> size should be a dictionary on of the following set of keys: ({'height', 'width'}, {'shortest_edge'}, {'longest_edge', 'shortest_edge'}, {'longest_edge'}), got 224. Converted to {'height': 224, 'width': 224}.
[INFO|image_processing_utils.py:424] 2024-06-02 00:04:26,400 >> Image processor LayoutLMv3ImageProcessor {
  "_valid_processor_keys": [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "apply_ocr",
    "ocr_lang",
    "tesseract_config",
    "return_tensors",
    "data_format",
    "input_data_format"
  ],
  "apply_ocr": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "LayoutLMv3ImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "ocr_lang": null,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 224,
    "width": 224
  },
  "tesseract_config": ""
}

[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,806 >> loading file vocab.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/vocab.json
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,807 >> loading file merges.txt from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/merges.txt
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,807 >> loading file tokenizer.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,807 >> loading file added_tokens.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,807 >> loading file special_tokens_map.json from cache at None
[INFO|tokenization_utils_base.py:2108] 2024-06-02 00:04:26,807 >> loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--microsoft--layoutlmv3-base/snapshots/cfbbbff0762e6aab37086fdd4739ad14fe7d5db4/tokenizer_config.json
[INFO|processing_utils.py:400] 2024-06-02 00:04:27,475 >> Processor LayoutLMv3Processor:
- image_processor: LayoutLMv3ImageProcessor {
  "_valid_processor_keys": [
    "images",
    "do_resize",
    "size",
    "resample",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "apply_ocr",
    "ocr_lang",
    "tesseract_config",
    "return_tensors",
    "data_format",
    "input_data_format"
  ],
  "apply_ocr": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.5,
    0.5,
    0.5
  ],
  "image_processor_type": "LayoutLMv3ImageProcessor",
  "image_std": [
    0.5,
    0.5,
    0.5
  ],
  "ocr_lang": null,
  "resample": 2,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "height": 224,
    "width": 224
  },
  "tesseract_config": ""
}

- tokenizer: LayoutLMv3TokenizerFast(name_or_path='microsoft/layoutlmv3-base', vocab_size=50265, model_max_length=512, is_fast=True, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={
	0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	50264: AddedToken("<mask>", rstrip=False, lstrip=True, single_word=False, normalized=True, special=True),
}

{
  "processor_class": "LayoutLMv3Processor"
}

06/02/2024 00:04:27 - INFO - llamafactory.data.loader - Loading dataset mllm_demo.json...
num_proc must be <= 6. Reducing num_proc to 6 for dataset of size 6.
Converting format of dataset (num_proc=6): 100%|█| 6/6 [00:00<00:00, 52.17 examp
num_proc must be <= 6. Reducing num_proc to 6 for dataset of size 6.
Running tokenizer on dataset (num_proc=6):   0%|   | 0/6 [00:00<?, ? examples/s]
multiprocess.pool.RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/multiprocess/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/utils/py_utils.py", line 623, in _write_generator_to_queue
    for i, result in enumerate(func(**kwargs)):
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3482, in _map_single
    batch = apply_function_on_filtered_inputs(
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3361, in apply_function_on_filtered_inputs
    processed_inputs = function(*fn_args, *additional_args, **fn_kwargs)
  File "/root/LLaMA-Factory/src/llamafactory/data/processors/supervised.py", line 51, in preprocess_supervised_dataset
    template.encode_multiturn(
  File "/root/LLaMA-Factory/src/llamafactory/data/template.py", line 66, in encode_multiturn
    return self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)
  File "/root/LLaMA-Factory/src/llamafactory/data/template.py", line 103, in _encode
    encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))
  File "/root/LLaMA-Factory/src/llamafactory/data/template.py", line 117, in _convert_elements_to_ids
    token_ids += tokenizer.encode(elem, add_special_tokens=False)
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/tokenization_utils_base.py", line 2654, in encode
    encoded_inputs = self.encode_plus(
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py", line 471, in encode_plus
    return self._encode_plus(
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py", line 684, in _encode_plus
    batched_output = self._batch_encode_plus(
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/transformers/models/layoutlmv3/tokenization_layoutlmv3_fast.py", line 533, in _batch_encode_plus
    encodings = self._tokenizer.encode_batch(
TypeError: PreTokenizedEncodeInput must be Union[PreTokenizedInputSequence, Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence]]
"""

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/root/anaconda3/envs/huggingface/bin/llamafactory-cli", line 8, in <module>
    sys.exit(main())
  File "/root/LLaMA-Factory/src/llamafactory/cli.py", line 98, in main
    run_exp()
  File "/root/LLaMA-Factory/src/llamafactory/train/tuner.py", line 33, in run_exp
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
  File "/root/LLaMA-Factory/src/llamafactory/train/sft/workflow.py", line 33, in run_sft
    dataset = get_dataset(model_args, data_args, training_args, stage="sft", **tokenizer_module)
  File "/root/LLaMA-Factory/src/llamafactory/data/loader.py", line 176, in get_dataset
    dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names, **kwargs)
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 593, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 558, in wrapper
    out: Union["Dataset", "DatasetDict"] = func(self, *args, **kwargs)
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/arrow_dataset.py", line 3197, in map
    for rank, done, content in iflatmap_unordered(
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/utils/py_utils.py", line 663, in iflatmap_unordered
    [async_result.get(timeout=0.05) for async_result in async_results]
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/datasets/utils/py_utils.py", line 663, in <listcomp>
    [async_result.get(timeout=0.05) for async_result in async_results]
  File "/root/anaconda3/envs/huggingface/lib/python3.10/site-packages/multiprocess/pool.py", line 774, in get
    raise self._value
TypeError: PreTokenizedEncodeInput must be Union[PreTokenizedInputSequence, Tuple[PreTokenizedInputSequence, PreTokenizedInputSequence]]
```