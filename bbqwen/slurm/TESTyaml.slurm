`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]Loading checkpoint shards:  50%|█████     | 1/2 [00:03<00:03,  3.64s/it]Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  1.74s/it]Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00,  2.03s/it]
Some weights of BBQwen2VLForConditionalGeneration were not initialized from the model checkpoint at Qwen/Qwen2-VL-2B-Instruct and are newly initialized: ['bbox_head.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Loaded annotations size: 50000
Line bbox format: [60, 4, 1036, 48]
Word bbox format: [60, 5, 150, 41]
Word bbox format: [158, 4, 197, 43]
Word bbox format: [217, 4, 268, 38]
Word bbox format: [275, 4, 365, 40]
Word bbox format: [374, 4, 396, 48]
Word bbox format: [418, 4, 520, 45]
Word bbox format: [534, 4, 589, 41]
Word bbox format: [610, 5, 644, 39]
Word bbox format: [663, 4, 757, 42]
Final dataset size: 10
Data range used: (0, 10)
Loaded annotations size: 50000
Word bbox format: [809, 6, 933, 42]
Word bbox format: [954, 6, 1036, 47]
Line bbox format: [250, 6, 1029, 59]
Word bbox format: [250, 7, 532, 59]
Word bbox format: [558, 6, 747, 54]
Word bbox format: [763, 7, 886, 56]
Word bbox format: [905, 7, 1029, 57]
Line bbox format: [236, 12, 1039, 59]
Word bbox format: [236, 12, 414, 48]
Word bbox format: [427, 14, 562, 59]
Final dataset size: 10
Data range used: (11, 21)
/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/training_args.py:1568: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
  warnings.warn(
  0%|          | 0/3 [00:00<?, ?it/s]__getitem__ called with idx: 2
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [158, 4, 197, 43]
text value: in
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: in
Input bbox: [158, 4, 197, 43]
Processed bbox: [[158, 4, 197, 43]]
Split words: ['in']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.1371527777777778, 0.0625, 0.17100694444444445, 0.671875]]
Word: in, Tokens: ['in']
All tokens: ['in']
Input ids: [258]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 6
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [418, 4, 520, 45]
text value: After
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: After
Input bbox: [418, 4, 520, 45]
Processed bbox: [[418, 4, 520, 45]]
Split words: ['After']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.3628472222222222, 0.0625, 0.4513888888888889, 0.703125]]
Word: After, Tokens: ['After']
All tokens: ['After']
Input ids: [6025]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 1
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [60, 5, 150, 41]
text value: goals
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: goals
Input bbox: [60, 5, 150, 41]
Processed bbox: [[60, 5, 150, 41]]
Split words: ['goals']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.052083333333333336, 0.078125, 0.13020833333333334, 0.640625]]
Word: goals, Tokens: ['goals']
All tokens: ['goals']
Input ids: [84157]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 8
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [610, 5, 644, 39]
text value: LA
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: LA
Input bbox: [610, 5, 644, 39]
Processed bbox: [[610, 5, 644, 39]]
Split words: ['LA']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.5295138888888888, 0.078125, 0.5590277777777778, 0.609375]]
Word: LA, Tokens: ['LA']
All tokens: ['LA']
Input ids: [17845]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 4
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [275, 4, 365, 40]
text value: years
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: years
Input bbox: [275, 4, 365, 40]
Processed bbox: [[275, 4, 365, 40]]
Split words: ['years']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.2387152777777778, 0.0625, 0.3168402777777778, 0.625]]
Word: years, Tokens: ['years']
All tokens: ['years']
Input ids: [41720]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 5
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [374, 4, 396, 48]
text value: -
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: -
Input bbox: [374, 4, 396, 48]
Processed bbox: [[374, 4, 396, 48]]
Split words: ['-']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.3246527777777778, 0.0625, 0.34375, 0.75]]
Word: -, Tokens: ['-']
All tokens: ['-']
Input ids: [12]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 0
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [60, 4, 1036, 48]
text value: goals in the years - After the LA Games he retired from
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: goals in the years - After the LA Games he retired from
Input bbox: [60, 4, 1036, 48]
Processed bbox: [[60, 4, 1036, 48]]
Split words: ['goals', 'in', 'the', 'years', '-', 'After', 'the', 'LA', 'Games', 'he', 'retired', 'from']
Words count: 12, Bbox count: 12
Normalized boxes: [[0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75], [0.052083333333333336, 0.0625, 0.8993055555555556, 0.75]]
Word: goals, Tokens: ['goals']
Word: in, Tokens: ['in']
Word: the, Tokens: ['the']
Word: years, Tokens: ['years']
Word: -, Tokens: ['-']
Word: After, Tokens: ['After']
Word: the, Tokens: ['the']
Word: LA, Tokens: ['LA']
Word: Games, Tokens: ['Games']
Word: he, Tokens: ['he']
Word: retired, Tokens: ['ret', 'ired']
Word: from, Tokens: ['from']
All tokens: ['goals', 'in', 'the', 'years', '-', 'After', 'the', 'LA', 'Games', 'he', 'ret', 'ired', 'from']
Input ids: [84157, 258, 1782, 41720, 12, 6025, 1782, 17845, 35807, 383, 2122, 2690, 1499]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 9
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [663, 4, 757, 42]
text value: Games
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: Games
Input bbox: [663, 4, 757, 42]
Processed bbox: [[663, 4, 757, 42]]
Split words: ['Games']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.5755208333333334, 0.0625, 0.6571180555555556, 0.65625]]
Word: Games, Tokens: ['Games']
All tokens: ['Games']
Input ids: [35807]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 3
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [217, 4, 268, 38]
text value: the
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: the
Input bbox: [217, 4, 268, 38]
Processed bbox: [[217, 4, 268, 38]]
Split words: ['the']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.18836805555555555, 0.0625, 0.2326388888888889, 0.59375]]
Word: the, Tokens: ['the']
All tokens: ['the']
Input ids: [1782]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
__getitem__ called with idx: 7
Current dataset length: 10
bbox type: <class 'list'>
bbox value: [534, 4, 589, 41]
text value: the
image size: (1152, 64)
Tokenizer type: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
Has cls_token: True
Has sep_token: True
Has pad_token: True
Input text: the
Input bbox: [534, 4, 589, 41]
Processed bbox: [[534, 4, 589, 41]]
Split words: ['the']
Words count: 1, Bbox count: 1
Normalized boxes: [[0.4635416666666667, 0.0625, 0.5112847222222222, 0.640625]]
Word: the, Tokens: ['the']
All tokens: ['the']
Input ids: [1782]
Final input_ids length: 512
Final attention_mask length: 512
Final bbox_tensors length: 512
COMPUTE LOSS CALLED

Inputs {'pixel_values': tensor([[[1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         ...,
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459]],

        [[1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         ...,
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459]],

        [[1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         ...,
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459]],

        [[1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         ...,
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459],
         [1.9303, 1.9303, 1.9303,  ..., 2.1459, 2.1459, 2.1459]]],
       device='cuda:0'), 'input_ids': tensor([[   258, 151643, 151643,  ..., 151643, 151643, 151643],
        [  6025, 151643, 151643,  ..., 151643, 151643, 151643],
        [ 84157, 151643, 151643,  ..., 151643, 151643, 151643],
        [ 17845, 151643, 151643,  ..., 151643, 151643, 151643]],
       device='cuda:0'), 'attention_mask': tensor([[1, 0, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0]], device='cuda:0')}
pixel_values shape: torch.Size([4, 328, 1176])
Traceback (most recent call last):
  File "/grphome/grp_handwriting/qwen/bbqwen/finetune_bb.py", line 242, in <module>
    main()
  File "/grphome/grp_handwriting/qwen/bbqwen/finetune_bb.py", line 236, in main
    trainer.train()
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/trainer.py", line 2123, in train
    return inner_training_loop(
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/trainer.py", line 2481, in _inner_training_loop
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/trainer.py", line 3579, in training_step
    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
  File "/grphome/grp_handwriting/qwen/bbqwen/finetune_bb.py", line 172, in compute_loss
    outputs = model(
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/accelerate/utils/operations.py", line 820, in forward
    return model_forward(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/accelerate/utils/operations.py", line 808, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/torch/amp/autocast_mode.py", line 43, in decorate_autocast
    return func(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/bbqwen/modeling_bbqwen2_vl.py", line 121, in forward
    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1041, in forward
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
  File "/grphome/grp_handwriting/qwen/qwenenv/lib/python3.8/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py", line 1034, in rot_pos_emb
    max_grid_size = grid_thw[:, 1:].max()
TypeError: list indices must be integers or slices, not tuple
  0%|          | 0/3 [00:07<?, ?it/s]
