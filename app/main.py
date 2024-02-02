import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline

HF_MODEL_ID = os.getenv("HF_MODEL_ID", "textminr/tl-flan-t5-base-onnx")
HF_TOKENIZER_ID = os.getenv("HF_TOKENIZER_ID", HF_MODEL_ID)
pipe = None

class ModelInput(BaseModel):
    top_terms: str

@asynccontextmanager
async def pipe_lifespan(app: FastAPI):
    global pipe

    model = ORTModelForSeq2SeqLM.from_pretrained(HF_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(HF_TOKENIZER_ID)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    yield

app = FastAPI(lifespan=pipe_lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/label_topic/")
def label_topic(model_input: ModelInput):
    return pipe(model_input.top_terms)