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
import uuid
import logging
import os
import time
from threading import Thread
import asyncio
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


def generate_uuid() -> str:
    return str(uuid.uuid4())


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    else:
        return 1


def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)


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


def get_prompt(messages: list):
    system_prompt = ''
    if len(messages) > 0 and messages[0]['role'] == 'system':
        system_prompt = messages[0]['content']

    sep = " "
    sep2 = " </s><s>"
    system_template = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    seps = [sep, sep2]
    if system_prompt.strip() != "":
        ret = system_template
    else:
        ret = "<s>[INST]"
    for i, chat in enumerate(messages):
        message = chat["content"]
        role = chat["role"]
        if i <= 1:
            if role == 'user':
                ret += " Human: " + message + " "
            elif role == 'system':
                pass
        else:
            if role == "user":
                ret += "[INST] Human:" + " " + message + seps[i % 2]
            else:
                ret += "[/INST] Assistant:" + " " + message + seps[i % 2]
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

    async def process_thread(self, body):
        json_obj = json.loads(body)
        messages = json_obj.get('messages')
        max_new_tokens = json_obj.get('max_tokens') or 2048
        top_p = json_obj.get('n') or 1
        temperature = json_obj.get('temperature') or 1

        prompt = get_prompt(messages)
        inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
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
        uid = generate_uuid()
        created = int(datetime.datetime.now().timestamp())
        for new_text in streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            if new_text != '</s>':
                message = {"id": uid, "object": "chat.completion.chunk",
                           "created": created, "model": "Cogen", "system_fingerprint": "",
                           "choices": [{"index": 0, "delta": {"role": "assistant", "content": new_text},
                                        "logprobs": None, "finish_reason": None}]}
                self.write('data:{}\n\n'.format(json.dumps(message)))
                await self.flush()
                bot_message += new_text
            await asyncio.sleep(0)
        end_time = time.time()
        self.write('data: [DONE]\n\n')
        await self.flush()
        print('生成耗时：', end_time - start_time, '文字长度：', len(bot_message), '字耗时：',
              (end_time - start_time) / len(bot_message))

    async def post(self):
        logging.info('new post request connect')
        body = self.request.body.decode('utf-8')
        logging.debug(f'request body {body}')
        try:
            await self.process_thread(body)
        except Exception as e:
            logging.warning(f'exception occur {repr(e)}')
            self.write('data: [DONE]\n\n')
            await self.flush()


class CogenPromptCompleteRequest(tornado.web.RequestHandler):

    def __init__(self, *args, **kwargs):
        super(CogenPromptCompleteRequest, self).__init__(*args, **kwargs)
        self.set_header('Content-Type', 'text/event-stream')
        self.set_header('Access-Control-Allow-Origin', "*")
        self.set_header("Access-Control-Allow-Headers", "*")
        # 请求方式
        self.set_header("Access-Control-Allow-Methods", "*")

    async def post(self):
        logging.info('new post request connect')
        body = self.request.body.decode('utf-8')
        json_obj = json.loads(body)
        prompt = json_obj.get('prompt')
        messages = [{"role": "system",
                     "content": "You are a helpful prompt engineer.Creates a completion for the provided prompt"},
                    {"role": "user", "content": prompt}]
        max_new_tokens = json_obj.get('max_tokens') or 2048
        temperature = json_obj.get('temperature') or 1

        prompt = get_prompt(messages)
        inputs = tokenizer([prompt], return_tensors='pt').to("cuda")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=1,
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
        uid = generate_uuid()
        created = int(datetime.datetime.now().timestamp())
        async for new_text in streamer:
            print(new_text, end='', flush=True)
            if len(new_text) == 0:
                continue
            if new_text != '</s>':
                message = {
                    "id": uid,
                    "object": "text_completion",
                    "created": created,
                    "choices": [
                        {
                            "text": new_text,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None
                        }
                    ],
                    "model": "Cogen",
                    "system_fingerprint": ""
                }

                self.write('data:{}\n\n'.format(json.dumps(message)))
                await self.flush()
        end_time = time.time()
        self.write('data: [DONE]\n\n')
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
            self.write('data:{"code": 200, "msg": "success", "data": "text {idx}"}\n\n')
            await self.flush()
            time.sleep(3)


def make_app():
    return tornado.web.Application([
        (r"/cogen/v1/chat/completions", CogenChatRequest),
        (r"/cogen//v1/completions", CogenPromptCompleteRequest),
        (r"/cogen/v1/test", TestChatRequest)
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    app = make_app()

    app.listen(6006)
    logging.info("sse服务启动")
    tornado.ioloop.IOLoop.current().start()
