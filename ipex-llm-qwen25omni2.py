from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from ipex_llm import optimize_model
from qwen_omni_utils import process_mm_info
import time
import argparse

model_path = "Qwen/Qwen2.5-Omni-7B"

parser = argparse.ArgumentParser(description=f"Predict Tokens using generate() API for {model_path}")
parser.add_argument("video_path", type=str, help="Path to the video file")
parser.add_argument('--low-bit', type=str,
    default='sym_int4' ,
    help='load_in_low_bit, "float" to not use low bit, other options are sym_int4, asym_int4, sym_int5, asym_int5, sym_int8,nf3,nf4, fp4, fp8, fp8_e4m3, fp8_e5m2, fp6, gguf_iq2_xxs, gguf_iq2_xs, gguf_iq1_s, gguf_q4k_m, gguf_q4k_s,fp16, fp6_k, seeing https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Overview/KeyFeatures/optimize_model.md')
parser.add_argument('--prompt', type=str, help="optional text prompt")
args = parser.parse_args()

model = Qwen2_5OmniModel.from_pretrained(model_path, enable_audio_output=False)
model = optimize_model(model, low_bit=args.low_bit,
                   modules_to_not_convert=["audio_tower", "visual", "token2wav"])
model = model.half().to('xpu')

processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

# video input (use audio in video)
conversation = [
    {
        "role": "system",
        "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": args.video_path},
        ],
    },
]

if (args.prompt):
    conversation[1]["content"].append({"type": "text", "text": args.prompt})

# image input
# conversation = [
#     {
#         "role": "system",
#         "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": r"test.png"},
#             {"type": "text", "text": "Describe the image in detail"},
#         ],
#     },
# ]

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
inputs = processor(text=text, audios=audios, images=images, videos=videos, return_tensors="pt", padding=True)
inputs = inputs.to(model.device).to(model.dtype)

for _ in range(3):
    start_time = time.time()
    # note: use `thinker_max_new_tokens` instead of `max_new_tokens`
    text_ids = model.generate(**inputs, use_audio_in_video=True, thinker_max_new_tokens=128)
    end_time = time.time()
    print(f"-- generate time = {end_time - start_time:.2f} s")
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
