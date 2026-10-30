#!/usr/bin/env python3
import argparse
import json, os, re, time
from pathlib import Path

os.environ.setdefault("UNSLOTH_DISABLE_TORCH_COMPILE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn.functional as F

def backend_tokenizer(t): return t.tokenizer if hasattr(t,'tokenizer') else t

def render(tokenizer,row,maxlen):
    messages=row['messages']; tools=row['tools']
    kw={'tools':tools,'enable_thinking':False,'tokenize':False}
    full=tokenizer.apply_chat_template(messages,add_generation_prompt=False,**kw)
    prefix=tokenizer.apply_chat_template(messages[:-1],add_generation_prompt=True,**kw)
    if not full.startswith(prefix): raise ValueError('prefix text mismatch')
    tok=backend_tokenizer(tokenizer)
    ids=tok.encode(full,add_special_tokens=False); pids=tok.encode(prefix,add_special_tokens=False)
    if ids[:len(pids)]!=pids: raise ValueError('prefix token mismatch')
    if len(ids)>maxlen: raise ValueError(f'length {len(ids)} > {maxlen}')
    raw=ids[len(pids):]; target_text=full[len(prefix):]
    m=re.match(r'^<think>.*?</think>\s*',target_text,flags=re.S)
    skip=0
    if m:
        skip_ids=tok.encode(target_text[:m.end()],add_special_tokens=False)
        if raw[:len(skip_ids)]!=skip_ids: raise ValueError('think mask mismatch')
        skip=len(skip_ids)
    target=raw[skip:]
    if not target: raise ValueError('empty target')
    return torch.tensor(ids,dtype=torch.long),torch.tensor(target,dtype=torch.long),len(pids)+skip

def logits_for_target(model,ids,n):
    out=model(input_ids=ids.unsqueeze(0),use_cache=False,logits_to_keep=n+1).logits
    return out[:,-(n+1):-1,:].squeeze(0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--max-seq-length',type=int,default=20000)
    args=parser.parse_args()
    data=Path(args.data).expanduser().resolve()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    output=Path(args.output).expanduser().resolve()
    local=int(os.environ.get('LOCAL_RANK','0')); rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1'))
    torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl')
    output.mkdir(parents=True,exist_ok=True)
    from peft import PeftModel
    from utils import load_model_and_tokenizer_unsloth
    model,tokenizer=load_model_and_tokenizer_unsloth('qwen','9b',max_seq_length=args.max_seq_length,load_in_4bit=True,full_finetuning=False)
    model=PeftModel.from_pretrained(model,checkpoint,is_trainable=False)
    model.eval(); model.config.use_cache=False
    rows=[json.loads(x) for x in data.open(encoding='utf-8') if x.strip()]
    sample_sum=0.0; token_nll=0.0; tokens=0; count=0; errors=[]; start=time.time()
    progress=output/f'progress_rank{rank}.json'
    with torch.inference_mode():
      for idx in range(rank,len(rows),world):
        try:
          ids,target,_=render(tokenizer,rows[idx],args.max_seq_length); ids=ids.cuda(); target=target.cuda()
          logits=logits_for_target(model,ids,len(target)).float()
          mean=F.cross_entropy(logits,target,reduction='mean')
          total=F.cross_entropy(logits,target,reduction='sum')
          sample_sum+=float(mean); token_nll+=float(total); tokens+=len(target); count+=1
        except Exception as e: errors.append({'index':idx,'error':repr(e)})
        if count%10==0:
          progress.write_text(json.dumps({'rank':rank,'completed':count,'assigned':len(range(rank,len(rows),world)),'sample_mean':sample_sum/max(count,1),'tokens':tokens,'errors':errors,'elapsed_seconds':time.time()-start},ensure_ascii=False,indent=2)+'\n')
    vals=torch.tensor([sample_sum,token_nll,tokens,count,len(errors)],device='cuda',dtype=torch.float64)
    if world>1: dist.all_reduce(vals,op=dist.ReduceOp.SUM)
    if rank==0:
      a,b,c,d,e=vals.cpu().tolist()
      result={'checkpoint':str(checkpoint),'valid_file':str(data),'valid_rows':len(rows),'evaluated_rows':int(d),'error_rows':int(e),'eval_loss_sample_mean':a/d,'eval_loss_token_weighted':b/c,'supervised_tokens':int(c),'world_size':world,'elapsed_seconds':time.time()-start,'mask_policy':'prefix and complete leading <think> block masked; loss begins at assistant text or tool call'}
      (output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
      print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if world>1: dist.destroy_process_group()

if __name__=='__main__': main()
