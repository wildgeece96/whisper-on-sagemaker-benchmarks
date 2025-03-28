"""This inference code is also based on
https://github.com/aws-samples/amazon-sagemaker-host-and-inference-whisper-model/blob/main/huggingface/code/inference.py
"""

import os
import types
import logging
import io
import tempfile
import json
import time

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,BaseModelOutput
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperConfig
from transformers import pipeline
from datasets import load_dataset

MODEL_ID = "openai/whisper-large-v3"


def model_fn(model_dir):
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    config = WhisperConfig.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration(config)

    def enc_f(self, input_features, attention_mask, **kwargs):
        if hasattr(self, 'forward_neuron'):
            out = self.forward_neuron(input_features, attention_mask)
        else:
            out = self.forward_(input_features, attention_mask, return_dict=True)
        return BaseModelOutput(**out)

    def dec_f(self, input_ids, attention_mask=None, encoder_hidden_states=None, **kwargs):
        output = None        
        if not attention_mask is None and encoder_hidden_states is None:
            # this is a workaround to align the input parameters for NeuronSDK tracer
            # None values are not allowed during compilation
            encoder_hidden_states, attention_mask = attention_mask,encoder_hidden_states
        inputs = [input_ids, encoder_hidden_states]
        
        # pad the input to max_dec_len
        if inputs[0].shape[1] > self.max_length:
            raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
        pad_size = torch.as_tensor(self.max_length - inputs[0].shape[1])
        inputs[0] = F.pad(inputs[0], (0, pad_size), "constant", processor.tokenizer.pad_token_id)
        
        if hasattr(self, 'forward_neuron'):
            output = self.forward_neuron(*inputs)
        else:
            # output_attentions is required if you want timestamps
            output = self.forward_(input_ids=inputs[0], encoder_hidden_states=inputs[1], return_dict=True, use_cache=False, output_attentions=output_attentions)
        # unpad the output
        output['last_hidden_state'] = output['last_hidden_state'][:, :input_ids.shape[1], :]
        # neuron compiler doesn't like tuples as values of dicts, so we stack them into tensors
        # also, we need to average axis=2 given we're not using cache (use_cache=False)
        # that way, to avoid an issue with the pipeline we change the shape from:
        #  bs,num selected,num_tokens,1500 --> bs,1,num_tokens,1500
        # I suspect there is a bug in the HF pipeline code that doesn't support use_cache=False for
        # word timestamps, that's why we need that.
        if not output.get('attentions') is None:
            output['attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :input_ids.shape[1]], axis=2, keepdim=True) for o in output['attentions']])
        if not output.get('cross_attentions') is None:
            output['cross_attentions'] = torch.stack([torch.mean(o[:, :, :input_ids.shape[1], :], axis=2, keepdim=True) for o in output['cross_attentions']])
        return BaseModelOutputWithPastAndCrossAttentions(**output)

    def proj_out_f(self, inp):
        pad_size = torch.as_tensor(self.max_length - inp.shape[1], device=inp.device)
        # pad the input to max_dec_len
        if inp.shape[1] > self.max_length:
            raise Exception(f"The decoded sequence is not supported. Max: {self.max_length}")
        x = F.pad(inp, (0,0,0,pad_size), "constant", processor.tokenizer.pad_token_id)
        
        if hasattr(self, 'forward_neuron'):
            out = self.forward_neuron(x)
        else:
            out = self.forward_(x)
        # unpad the output before returning
        out = out[:, :inp.shape[1], :]
        return out

    if not hasattr(model.model.encoder, 'forward_'): model.model.encoder.forward_ = model.model.encoder.forward
    if not hasattr(model.model.decoder, 'forward_'): model.model.decoder.forward_ = model.model.decoder.forward
    if not hasattr(model.proj_out, 'forward_'): model.proj_out.forward_ = model.proj_out.forward

    model.model.encoder.forward = types.MethodType(enc_f, model.model.encoder)
    model.model.decoder.forward = types.MethodType(dec_f, model.model.decoder)
    model.proj_out.forward = types.MethodType(proj_out_f, model.proj_out)
    max_dec_len = 128
    model.model.decoder.max_length = max_dec_len
    model.proj_out.max_length = max_dec_len

    model.model.encoder.forward_neuron = torch.jit.load(
        os.path.join(model_dir, "whisper_large-v3_1_neuron_encoder.pt")
        )
    model.model.decoder.forward_neuron = torch.jit.load(
        os.path.join(model_dir, "whisper_large-v3_1_128_neuron_decoder.pt")
    )
    model.proj_out.forward_neuron = torch.jit.load(
        os.path.join(model_dir, "whisper_large-v3_1_128_neuron_proj.pt")
    )

    # warmpup whisper
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # sample #3 is ~9.9seconds and produces 33 output tokens + pad token
    sample = dataset[3]["audio"]
    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
    model.generate(input_features)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_ID,
        chunk_length_s=30
    )
    pipe.model = model
    return pipe


def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
     
    logging.info("Check out the request_body type: %s", type(request_body))
    
    start_time = time.time()
    
    file = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    logging.info("Start to generate the transcription ...")
    result = model(tfile.name, batch_size=1)["text"]
    
    logging.info("Upload transcription results back to S3 bucket ...")
    
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("The time for running this program is %s", elapsed_time)
    
    return json.dumps(result), response_content_type   

