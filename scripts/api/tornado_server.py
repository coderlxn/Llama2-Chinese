import binascii
import random
from typing import Optional, Awaitable, Any
import tornado.ioloop
import tornado.web
import datetime
import multiprocessing as mp
import httpx
import base64
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


class CogenChatRequest(tornado.web.RequestHandler):

    def __init__(self, *args, **kwargs):
        super(CogenChatRequest, self).__init__(*args, **kwargs)
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Access-Control-Allow-Origin', "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        # 请求方式
        self.set_header("Access-Control-Allow-Methods", "*")

    async def post(self):
        logging.info('new post request connect')
        global model, tokenizer
        body = self.request.body.decode('utf-8')
        json_post_list = json.loads(body)
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

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        start_time = time.time()
        bot_message = ''
        print('Human:', prompt)
        print('Assistant: ', end='', flush=True)
        for new_text in streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            if new_text != '</s>':
                bot_message += new_text
            if 'Human:' in bot_message:
                bot_message = bot_message.split('Human:')[0]
            # history[-1][1] = bot_message
            print(f'bot message from streamer {bot_message}')
            self.write(f'data:{"code": 200, "msg": "success", "data": "text {bot_message}"}\n\n')
            await self.flush()
        end_time = time.time()
        self.write('data:{"code": 200, "msg": "done", "data": {}}\n\n')
        print('生成耗时：', end_time - start_time, '文字长度：', len(bot_message), '字耗时：',
              (end_time - start_time) / len(bot_message))


class TestChatRequest(tornado.web.RequestHandler):

    def __init__(self, *args, **kwargs):
        super(TestChatRequest, self).__init__(*args, **kwargs)
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Access-Control-Allow-Origin', "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        # 请求方式
        self.set_header("Access-Control-Allow-Methods", "*")

    async def post(self):
        logging.info('new post request connect')

        for idx in range(10):
            self.write(f'data:{"code": 200, "msg": "success", "data": "text {idx}"}\n\n')
            await self.flush()


def make_app():
    return tornado.web.Application([
        (r"/cogen/v1/chat", CogenChatRequest),
        (r"/cogen/v1/test", TestChatRequest)
    ])


if __name__ == "__main__":
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

    app = make_app()

    i18n_path = os.path.join(os.path.dirname(__file__), 'locales')
    tornado.locale.load_translations(i18n_path)
    tornado.locale.set_default_locale('en_US')

    app.listen(6006)
    logging.info("sse服务启动")
    tornado.ioloop.IOLoop.current().start()
