
# # Conditional Generation with Prefix Tuning.
# In this tutorial, we do conditional generation with prefix tuning template.

# we use WebNLG as an example, as well. Note that the evaluation of generation result should be done
# by using the scripts provided by https://github.com/Yale-LILY/dart/tree/master/evaluation,
# Which we do not include in it.

import inspect
import argparse
import torch
import logging
import numpy as np
import random

seed = 13
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
args = parser.parse_args()
print(args)

from openprompt.data_utils.conditional_generation_dataset import FLEURSProcessor
fleurs_path = "/mount/arbeitsdaten45/projekte/asr-4/denisopl/fleurs/"
dataset = {}
dataset['train'] = FLEURSProcessor().get_train_examples(fleurs_path)
dataset['validation'] = FLEURSProcessor().get_dev_examples(fleurs_path)
dataset['test'] = FLEURSProcessor().get_test_examples(fleurs_path)


# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"mask"}')
#mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])
print(wrapped_example)


batch_size = 32

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader
train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=batch_size, shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=batch_size,shuffle=False, teacher_forcing=False, predict_eos_token=True,
    truncate_method="head")

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model = prompt_model.cuda()


# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
{
    "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
    "weight_decay": 0.0,
},
{
    "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
    "weight_decay": 0.0,
},
]


optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

from transformers.optimization import get_linear_schedule_with_warmup

max_epoch = 10
tot_step  = len(train_dataloader) * max_epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []
    prompt_model.eval()

    for step, inputs in enumerate(dataloader):
        logging.info(f"Generation step {step}")
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "cer")
    return score


model_args = set(inspect.signature(prompt_model.prepare_inputs_for_generation).parameters)

if "kwargs" in model_args or "model_kwargs" in model_args:
    model_args |= set(inspect.signature(prompt_model.forward).parameters)


generation_arguments = {
    "max_length": None,
    "max_new_tokens": 256,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "model_args": model_args,
}

# training and generation.
global_step = 0
tot_loss = 0
log_loss = 0

best_val_loss = 99999
best_val_epoch = -1
best_state_dict = None
patience = 2

for epoch in range(max_epoch):
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        global_step += 1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()
        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step % 10 == 0:
            logging.info("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch+1, global_step, (tot_loss-log_loss)/ 10, scheduler.get_last_lr()[0]))
            log_loss = tot_loss

    prompt_model.eval()

    val_loss = 0
    val_step = 0
    
    for _, inputs in enumerate(validation_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        val_loss += loss.item()
        val_step += 1

    val_loss = val_loss / val_step
    logging.info(f"Validation loss for epoch {epoch}: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        best_state_dict = prompt_model.state_dict()
    elif epoch - best_val_epoch > patience:
        logging.info(f"Validation loss has not improved since epoch {best_val_epoch} (>{patience} epochs), stopping training")
        break

prompt_model.load_state_dict(best_state_dict)

score = evaluate(prompt_model, validation_dataloader)
logging.info(f"Validation CER: {score}")

score = evaluate(prompt_model, test_dataloader)
logging.info(f"Test CER: {score}")
