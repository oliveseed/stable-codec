# Stable Codec

This repository contains training and inference scripts for models in the Stable Codec series, starting with `stable-codec-speech-16k` - introduced in the paper titled Scaling Transformers for Low-bitrate High-Quality Speech Coding.

Paper: https://arxiv.org/abs/2411.19842

Sound demos: https://stability-ai.github.io/stable-codec-demo/

## Additional training

In addition to the training described in the paper, the released weights have also undergone 500k steps of finetuning with force-aligned phoneme data from LibriSpeech and the English portion Multilingual LibriSpeech. This was performed by using a CTC head to regress the phoneme categories from pre-bottleneck latents. We found that this additional training significantly boosted the applicability of the codec tokens to downstream tasks like TTS.

## Install

The model itself is defined in [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) package.

To install `stable-codec`:

```bash
pip install -e .
pip install -U flash-attn --no-build-isolation
```

**IMPORTANT NOTE:** This model currently has a hard requirement for FlashAttention due to its use of sliding window attention. Inference without FlashAttention will likely be greatly degraded. 

## Encoding and decoding

To encode audio or decode tokens, the `StableCodec` class provides a convenient wrapper for the model. It can be used with a local checkpoint and config as follows:

```python
from stable_codec.model import StableCodec

model = StableCodec(
    model_config_path="<path-to-model-config>",
    ckpt_path="<path-to-checkpoint>", # optional, can be `None`
)

audiopath = "audio.wav"

latents, tokens = model.encode(audiopath)
decoded_audio = model.decode(tokens)

torchaudio.save("decoded.wav", decoded_audio, model.sample_rate)
```

To download the model weights automatically from HuggingFace, simply provide the model name:

```python
model = StableCodec(
    pretrained_model = 'stabilityai/stable-codec-speech-16k'
)
```
### Posthoc bottleneck configuration

Most usecases will benefit from replacing the training-time FSQ bottleneck with a post-hoc FSQ bottleneck, as described in the paper. This allows token dictionary size to be reduced to a reasonable level for modern language models. This is achieved by calling the `set_posthoc_bottleneck` function, and setting a flag to the encode/decode calls:

```python
model.set_posthoc_bottleneck("2x15625_700bps")
latents, tokens = model.encode(audiopath, posthoc_bottleneck = True)
decoded_audio = model.decode(tokens, posthoc_bottleneck = True)
```
`set_posthoc_bottleneck` can take a string as argument, which allows selection a number of recommended preset settings for the bottleneck:

| Bottleneck Preset | Number of Tokens per step | Dictionary Size | Bits Per Second (bps) |
|-------------------|------------------|-----------------|-----------------------|
| `1x46656_400bps`   | 1             | 46656             | 400                   |
| `2x15625_700bps`   | 2             | 15625             | 700                   |
| `4x729_1000bps`    | 4             | 729               | 1000                  |

Alternatively, the bottleneck stages can be specified directly. The format for specifying this can be seen in the definition of the `StableCodec` class in `model.py`.

### Normalization

The model is trained with utterances normalized to -20 LUFS. The `encode` function applies this by default, but it can be disabled by setting `normalize = False` when calling the function. 

## Finetune

To finetune a model given its config and checkpoint, execute `train.py` file:

```bash
python train.py \
    --project "stable-codec" \
    --name "finetune" \
    --config-file "defaults.ini" \
    --save-dir "<ckpt-save-dir>" \
    --model-config "<path-to-config.json>" \
    --dataset-config "<dataset-config.json>" \
    --val-dataset-config "<dataset-config.json>" \
    --pretrained-ckpt-path "<pretrained-model-ckpt.ckpt>" \
    --ckpt-path "$CKPT_PATH" \
    --num-nodes $SLURM_JOB_NUM_NODES \
    --num-workers 16 --batch-size 10 --precision "16-mixed" \
    --checkpoint-every 10000 \
    --logger "wandb"
```

For dataset configuration, refer to `stable-audio-tools` [dataset docs](https://github.com/Stability-AI/stable-audio-tools/blob/main/docs/datasets.md).


### Using CTC loss

To use [CTC loss](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)
during training you have to enable it in the training configuration file
and in the training dataset configuration.

1. Modifying training configuration:
    - Enable CTC projection head and set its hidden dimension:
      ```python
      config["model"]["use_proj_head"] = True
      config["model"]["proj_head_dim"] = 81
      ```
    - Enable CTC in the training part of the config:
      ```python
      config["training"]["use_ctc"] = True
      ```
    - And set its loss config:
      ```python
      config["training"]["loss_configs"]["ctc"] = {
        "blank_idx": 80,
        "decay": 1.0,
        "weights": {"ctc": 1.0}
      }
      ```
    - Optionally, you can enable computation of the Phone-Error-Rate (PER) during validation:
      ```python
      config["training"]["eval_loss_configs"]["per"] = {}
      ```

2. Configuring dataset (only WebDataset format is supported for CTC):
   - The dataset configuration should have one additional field set to it (see [dataset docs](https://github.com/Stability-AI/stable-audio-tools/blob/main/docs/datasets.md) for other options):
     ```python
     config["force_align_text"] = True
     ```
   - And the JSON metadata file for each sample should contain force aligned transcript under `force_aligned_text` entry in the format specified below (besides other metadata).
     Where `transcript` is a list of word-level alignments with `start` and `end` fields specifying range **in seconds** of each word.
     ```json
     "normalized_text":"and i feel"
     "force_aligned_text":{
      "transcript":[
         {
            "word":"and",
            "start":0.2202,
            "end":0.3403
         },
         {
            "word":"i",
            "start":0.4604,
            "end":0.4804
         },
         {
            "word":"feel",
            "start":0.5204,
            "end":0.7006
         }
       ]
     }
     ```
