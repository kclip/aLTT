from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, StoppingCriteria, StoppingCriteriaList#, pipeline
import torch
from accelerate import Accelerator
import os
import gc
#from torch.nn.parallel import DistributedDataParallel as DDP
#import torch.distributed as dist

#TOKEN = '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'
cache_dir= '/scratch_tmp/prj/nmes_simeone/sangwoop/ARIA_pr/cache/'

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# print('CUDA: ', os.environ["CUDA_DEVICE_ORDER"])
#print(len(os.environ["CUDA_DEVICE_ORDER"]))
#print(torch.cuda.device_count())


def LLaMA_3_Instruct_device_map(num_layers=80):
    num_gpus = torch.cuda.device_count()
    num_layers_per_GPU = num_layers // num_gpus
    device_map = {}
    #per_layer_names = ['input_layernorm', 'mlp.down_proj', 'mlp.gate_proj', 'mlp.up_proj', 'post_attention_layernorm', 'self_attn.k_proj', 'self_attn.o_proj', 'self_attn.q_proj', 'self_attn.v_proj']
    for i in range(num_layers):
        gpu_for_i = i // (num_layers_per_GPU+1)
        #for name in per_layer_names:
            #device_map['model.layers.'+str(i)+ '.' + name + '.weight'] = gpu_for_i   
        device_map['model.layers.'+str(i)] = gpu_for_i
    device_map['model.embed_tokens.weight'] = 0 # to be the same with input!
    device_map['model.norm.weight'] = 0 #num_gpus-1
    device_map['lm_head.weight'] = 0 #num_gpus-1
    return device_map, num_gpus
#print(device_map)


class StopOnPunctuation(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # Ensure tokenizer is passed to avoid 'cls._tokenizer' issues

    def __call__(self, input_ids, scores, **kwargs):
        last_token_id = input_ids[0, -1].item()
        # Decode it and strip whitespace
        last_token = self.tokenizer.decode([last_token_id])
        # Check if any of the punctuation marks are present anywhere in the token
        # Check if the last character is a stopping punctuation mark
        return any(char in [".", "!", "?"] for char in last_token)



class HuggingFaceCompletion:
    _net = None
    _tokenizer = None
    _current_model_name = None
    _device= None
    _accelerator = None
    _e = None
    #"cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def create(cls, prompt, suffix='', model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.7, max_tokens=50,  top_p=1.0, frequency_penalty=0.0, n=1):
        if suffix != '':
            raise NotImplementedError
        # Load the model and tokenizer only once
        #if cls._net is None or cls._tokenizer is None:
        with torch.no_grad():
            if cls._net is None or cls._current_model_name != model:
                print(f"Loading model: {model}...")
                print('on the cache: ', cls._current_model_name, 'what we need now: ', model)
                if cls._net is None:
                    pass
                else:
                    cls._net.to('cpu')
                    del cls._net
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                cls._current_model_name = model
                cls._device = "cuda" if torch.cuda.is_available() else "cpu"

                device_map, num_gpus = LLaMA_3_Instruct_device_map(num_layers=80)
                with torch.no_grad():
                    if 't5' in model:
                        cls._net = AutoModelForSeq2SeqLM.from_pretrained(model, cache_dir=cache_dir).to(cls._device)
                    else:
                        cls._net = AutoModelForCausalLM.from_pretrained(model, device_map=device_map, torch_dtype="bfloat16", cache_dir=cache_dir)#.to(cls._device)
                        
                cls._net.eval()
                cls._tokenizer = AutoTokenizer.from_pretrained(model, cache_dir=cache_dir) #, return_token_type_ids=False
                cls._tokenizer.pad_token = cls._tokenizer.eos_token
                cls._tokenizer.pad_token_id = cls._tokenizer.eos_token_id
                cls._net.config.eos_token_id = cls._tokenizer.pad_token_id
                cls._net.config.pad_token_id = cls._net.config.eos_token_id

                #print(cls._net)
            #print('----------', cls._tokenizer.pad_token_id, cls._net.config.pad_token_id)
            inputs = cls._tokenizer(prompt, truncation=False, padding=True, return_tensors="pt", padding_side="left")
            device = next(cls._net.parameters()).device  # Get the device where the model is located
            #print('-----------------next device', device)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            # class StopOnPunctuation(StoppingCriteria):
            #     def __call__(self, input_ids, scores, **kwargs):
            #         # Stop if a period, exclamation, or question mark is generated
            #         last_token_id = input_ids[0, -1].item()
            #         last_token = cls._tokenizer.decode([last_token_id])
            #         return last_token.strip() in [".", "!", "?"]  # top
            # #         # last_token_id = input_ids[0, -1].item()
            # #         # decoded_token = cls._tokenizer.decode(last_token_id)
            # #         # return decoded_token in [".", "!", "?"]   # bottom 

            stopping_criteria = StopOnPunctuation(cls._tokenizer)
            outputs = cls._net.generate(
                #inputs.input_ids,
                inputs['input_ids'],
                #inputs_embeds = embeddings,
                attention_mask=inputs["attention_mask"], #.to(torch.bfloat16),
                # position_ids=position_ids_expanded,  # Pass it explicitly
                #max_length=inputs.input_ids.shape[1] + max_tokens,
                max_length=inputs['input_ids'].shape[1] + max_tokens,
                temperature=temperature,
                num_return_sequences=n,
                pad_token_id=cls._net.config.pad_token_id,
                stopping_criteria=[stopping_criteria],
                do_sample=True,  # Enable temperature sampling
                top_p=top_p,     # Apply top_p sampling
                repetition_penalty=frequency_penalty  # Apply frequency_penalty (repetition penalty)
            )

            #results = [cls._tokenizer.decode(output.tolist(), skip_special_tokens=True) for output in outputs]
            #results = [cls._tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
            #print('----------generated outputs')
            results = []
            for output in outputs:
                # Identify where the input prompt ends
                input_length = inputs['input_ids'].shape[1]  # Length of the original input prompt
                generated_tokens = output[input_length:]  # Exclude the input part
                text = cls._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                #print('---------------------original full generated text', text)
                #text = cls._tokenizer.decode(output, skip_special_tokens=True)
                # Search from the end of the text for the last punctuation mark
                for idx in range(len(text) - 1, -1, -1):
                    if text[idx] in [".", "!", "?"]:
                        text = text[:idx + 1]  # Keep the text up to the punctuation
                        break
                #print('---------------------after trimming the generated text', text)

                results.append(text)
            #print('----------postprocessed outputs')


            #print('results: ', results)
            del inputs, outputs
            # del cls._net
            gc.collect()
            # cls._net = None
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Mimicking OpenAI's response structure
            return {
                "choices": [{"text": result} for result in results]#,
                # "usage": {
                #     "prompt_tokens": inputs['input_ids'].numel(),
                #     "completion_tokens": sum(len(cls._tokenizer.encode(r)) for r in results),
                #     "total_tokens": inputs['input_ids'].numel() + sum(len(cls._tokenizer.encode(r)) for r in results)
                # }
            }
