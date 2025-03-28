
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,BaseModelOutput


def make_forward_neuron(processor, output_attentions=False):
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
    return enc_f, dec_f, proj_out_f