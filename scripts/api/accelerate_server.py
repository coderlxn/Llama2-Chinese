import argparse
import gc
import math
import logging
import os
import time
from threading import Thread

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch
import torch.distributed as dist

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--infer_dtype', default="int8", choices=["int4", "int8", "float16"], required=False, type=str)
parser.add_argument('--model_source', default="llama2_chinese", choices=["llama2_chinese", "llama2_meta"],
                    required=False, type=str)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = torch.cuda.device_count()

rank = local_rank

app = FastAPI()


def get_prompt_llama2chinese(
        chat_history, system_prompt: str
) -> str:
    prompt = ''
    for input_text_one in chat_history:
        prompt += "<s>" + input_text_one['role'] + ": " + input_text_one['content'].strip() + "\n</s>"
    if chat_history[-1]['role'] == 'Human':
        prompt += "<s>Assistant: "
    else:
        prompt += "<s>Human: "
    prompt = prompt[-2048:]
    if len(system_prompt) > 0:
        prompt = '<s>System: ' + system_prompt.strip() + '\n</s>' + prompt

    return prompt


def get_prompt(chat_history, system_prompt: str):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    sep = " "
    sep2 = " </s><s>"
    stop_token_ids = [2]
    system_template = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    roles = ("[INST]", "[/INST]")
    seps = [sep, sep2]
    if system_prompt.strip() != "":
        ret = system_template
    else:
        ret = "<s>[INST]"
    for i, chat in enumerate(chat_history):
        message = chat["content"]
        role = chat["role"]
        if message:
            if i == 0:
                ret += " Human: " + message + " "
            else:
                if role == "Human":
                    ret += "[INST] Human:" + " " + message + seps[i % 2]
                else:
                    ret += "[/INST] Assistant:" + " " + message + seps[i % 2]
        else:
            if role == "Human":
                ret += "[INST]"
            else:
                ret += "[/INST]"
    ret += "[/INST] Assistant:"
    print("prompt:{}".format(ret))
    return ret


async def stream_generate(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    history = json_post_list.get('history')
    system_prompt = json_post_list.get('system_prompt')
    max_new_tokens = json_post_list.get('max_new_tokens')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    prompt = get_prompt(history, system_prompt)
    inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=50,
        temperature=temperature,
        num_beams=1,
        repetition_penalty=1.2,
        max_length=2048,
    )

    # thread = Thread(target=model.generate, kwargs=generate_kwargs)
    # thread.start()
    # start_time = time.time()
    # bot_message = ''
    # print('Human:', prompt)
    # print('Assistant: ', end='', flush=True)
    # for new_text in streamer:
    #     print(new_text, end='', flush=True)
    #     if len(new_text) == 0:
    #         continue
    #     if new_text != '</s>':
    #         bot_message += new_text
    #     if 'Human:' in bot_message:
    #         bot_message = bot_message.split('Human:')[0]
    #     # history[-1][1] = bot_message
    #     print(f'bot message from streamer {bot_message}')
    #     yield bot_message
    # end_time = time.time()
    # print('生成耗时：', end_time - start_time, '文字长度：', len(bot_message), '字耗时：',
    #       (end_time - start_time) / len(bot_message))

    generate_ids = model.generate(**generate_kwargs)

    generate_ids = [item[len(inputs[0]):-1] for item in generate_ids]

    bot_message = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if 'Human:' in bot_message:
        bot_message = bot_message.split('Human:')[0]

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": bot_message,
        "status": 200,
        "time": time
    }
    return answer


@app.post("/generate")
async def create_item(request: Request):
    logging.info("generate function called")
    return StreamingResponse(stream_generate(request))


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


async def fake_video_streamer():
    for i in range(10):
        yield b"some fake video bytes"
        time.sleep(1)


@app.get("/test")
async def main_test():
    return StreamingResponse(fake_video_streamer())


if __name__ == '__main__':

    logging.basicConfig(level=logging.DEBUG)

    dtype = torch.float16
    kwargs = dict(
        device_map="auto",
    )
    print("get_world_size:{}".format(get_world_size()))

    infer_dtype = args.infer_dtype
    if infer_dtype not in ["int4", "int8", "float16"]:
        raise ValueError("infer_dtype must one of int4, int8 or float16")

    if get_world_size() > 1:
        kwargs["device_map"] = "balanced_low_0"

    if infer_dtype == "int8":
        print_rank0("Using `load_in_8bit=True` to use quanitized model")
        kwargs["load_in_8bit"] = True
    else:
        kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if infer_dtype in ["int8", "float16"]:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs)
    elif infer_dtype == "int4":
        from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model

        model = AutoGPTQForCausalLM.from_quantized(
            args.model_path, device="cuda:0",
            use_triton=False,
            low_cpu_mem_usage=True,
            # inject_fused_attention=False,
            # inject_fused_mlp=False
        )
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
