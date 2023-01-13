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
import sys

seed = 13
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-6)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default="bloom")  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default="bigscience/bloomz-560m")
parser.add_argument("--no_train", action="store_true")
parser.add_argument("--add_lang", choices=["none", "name", "code"], default="code")
parser.add_argument("--num_token", type=int, default=5)
parser.add_argument("--mid_dim", type=int, default=512)
parser.add_argument(
    "--task", choices=["langid_transcribe", "translate"], default="langid_transcribe"
)
parser.add_argument(
    "--text",
    type=str,
    default='Detect language and repeat the sentence: {"placeholder":"text_a"}. {"mask":None, "shortenable":True}',
)
args = parser.parse_args()
print(args)

dataset = {}

if args.task == "langid_transcribe":
    from openprompt.data_utils.conditional_generation_dataset import FLEURSProcessor

    processor = FLEURSProcessor(add_lang=args.add_lang)
    fleurs_path = "/mount/arbeitsdaten45/projekte/asr-4/denisopl/fleurs/"
    dataset["train"] = processor.get_train_examples(fleurs_path)
    dataset["validation"] = processor.get_dev_examples(fleurs_path)
    dataset["test"] = processor.get_test_examples(fleurs_path)
elif args.task == "translate":
    from openprompt.data_utils.conditional_generation_dataset import CoVoSTProcessor

    processor = CoVoSTProcessor()
    covost_path = "/mount/arbeitsdaten45/projekte/asr-4/denisopl/covost/data/"
    dataset["train"] = processor.get_train_examples(covost_path)
    dataset["validation"] = processor.get_dev_examples(covost_path)
    dataset["test"] = processor.get_test_examples(covost_path)

    args.text = 'Translate the following text {"placeholder":"text_a"} {"mask":None, "shortenable":True}'
    args.add_lang = "none"
else:
    print(f"Unknown task '{args.task}'")
    sys.exit()


# load a pretrained model, its tokenizer, its config, and its TokenzerWrapper by one function
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(
    args.model, args.model_name_or_path
)

# Instantiating the PrefixTuning Template !
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate

# we can use a plain text as the default setting
# i.e.
# mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer)
# is equal to
mytemplate = PrefixTuningTemplate(
    model=plm,
    tokenizer=tokenizer,
    text=args.text,
    num_token=args.num_token,
    mid_dim=args.mid_dim,
)
# mytemplate = PrefixTuningTemplate(model=plm,  tokenizer=tokenizer, text=' {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} ', using_decoder_past_key_values=False)

# To better understand how does the template wrap the example, we visualize one instance.
# You may observe that the example doesn't end with <|endoftext|> token. Don't worry, adding specific end-of-text token
# is a language-model-specific token. we will add it for you in the TokenizerWrapper once you pass `predict_eos_token=True`
wrapped_example = mytemplate.wrap_one_example(dataset["train"][0])
print(wrapped_example)

batch_size = 4

# Your can loop over the dataset by yourself by subsequently call mytemplate.wrap_one_example  and WrapperClass().tokenizer()
# but we have provide a PromptDataLoader for you.
from openprompt import PromptDataLoader

if args.no_train:
    train_loader = None
    validation_dataloader = None
    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=True,
        truncate_method="head",
    )

else:
    train_dataloader = PromptDataLoader(
        dataset=dataset["train"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=True,
        teacher_forcing=True,
        predict_eos_token=True,  # be sure to pass predict_eos_token=True if your template doesn't contain one, or you model may fail to stop generation.
        truncate_method="head",
    )

    validation_dataloader = PromptDataLoader(
        dataset=dataset["validation"],
        template=mytemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        decoder_max_length=256,
        batch_size=batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=True,
        truncate_method="head",
    )

    test_dataloader = None

# load the pipeline model PromptForGeneration.
from openprompt import PromptForGeneration

use_cuda = True
prompt_model = PromptForGeneration(
    plm=plm,
    template=mytemplate,
    freeze_plm=True,
    tokenizer=tokenizer,
    plm_eval_mode=args.plm_eval_mode,
)
if use_cuda:
    prompt_model = prompt_model.cuda()


# Follow PrefixTuning（https://github.com/XiangLi1999/PrefixTuning), we also fix the language model
# only include the template's parameters in training.

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in mytemplate.named_parameters()
            if (not any(nd in n for nd in no_decay)) and p.requires_grad
        ],
        "weight_decay": 0.0,
    },
    {
        "params": [
            p
            for n, p in mytemplate.named_parameters()
            if any(nd in n for nd in no_decay) and p.requires_grad
        ],
        "weight_decay": 0.0,
    },
]


def split_text(texts):
    langs, sentences = [[], []]

    for s in texts:
        lang, sentence = ["", ""]

        if args.add_lang == "name":
            if "language:" in s:
                parts = [x.strip() for x in s.split("language:", maxsplit=1)]
                lang, sentence = parts
            else:
                sentence = s
        elif args.add_lang == "code":
            if "]" in s:
                parts = [x.strip() for x in s.split("]", maxsplit=1)]
                lang, sentence = parts
            else:
                sentence = s
        else:
            sentence = s

        langs.append(lang)
        sentences.append(sentence)

    return langs, sentences


# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric

# Define evaluate function
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []

    generated_lang = []
    groundtruth_lang = []

    prompt_model.eval()

    with torch.no_grad():
        for step, inputs in enumerate(dataloader):
            logging.info(f"Generation step {step}")
            if use_cuda:
                inputs = inputs.cuda()
            _, output_sentence = prompt_model.generate(inputs, **generation_arguments)

            pred_lang, pred_sentence = split_text(output_sentence)
            generated_sentence.extend(pred_sentence)
            generated_lang.extend(pred_lang)

            tgt_lang, tgt_sentence = split_text(inputs["tgt_text"])
            groundtruth_sentence.extend(tgt_sentence)
            groundtruth_lang.extend(tgt_lang)

    cer = generation_metric(generated_sentence, groundtruth_sentence, "cer") * 100.0
    bleu = generation_metric(generated_sentence, groundtruth_sentence, "bleu")
    chrf = generation_metric(generated_sentence, groundtruth_sentence, "chrf")
    acc = (
        sum(1 for x, y in zip(generated_lang, groundtruth_lang) if x == y)
        / len(generated_lang)
        * 100.0
    )

    return f"Acc;CER;BLEU;CHRF: {acc:.02f};{cer:.02f};{bleu:.02f};{chrf:.02f}"


model_args = set(
    inspect.signature(prompt_model.prepare_inputs_for_generation).parameters
)

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
patience = 2
ckpt_file = f"{args.model_name_or_path}_{args.add_lang}_lr{args.lr}_b{batch_size}_t{args.num_token}_d{args.mid_dim}_l{len(args.text)}.bin".replace(
    "/", "-"
)

if not args.no_train:
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)

    from transformers.optimization import get_linear_schedule_with_warmup

    max_epoch = 20
    tot_step = len(train_dataloader) * max_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

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
                logging.info(
                    "Epoch {}, global_step {} average loss: {} lr: {}".format(
                        epoch + 1,
                        global_step,
                        (tot_loss - log_loss) / 10,
                        scheduler.get_last_lr()[0],
                    )
                )
                log_loss = tot_loss

        prompt_model.eval()

        val_loss = 0
        val_step = 0

        with torch.no_grad():
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
            torch.save(prompt_model.state_dict(), ckpt_file)
        elif epoch - best_val_epoch > patience:
            logging.info(
                f"Validation loss has not improved since epoch {best_val_epoch} (>{patience} epochs), stopping training"
            )
            break

if args.no_train:
    prompt_model.load_state_dict(torch.load(ckpt_file))

    #result = evaluate(prompt_model, validation_dataloader)
    #logging.info(f"Validation {result}")

    result = evaluate(prompt_model, test_dataloader)
    logging.info(f"Test {result}")
