import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (DataCollatorForLanguageModeling, GPT2Config,
                          GPT2LMHeadModel, GPT2TokenizerFast, Trainer,
                          TrainingArguments)

from args import config_args


@config_args
class Config:
    train: bool = False                     # train the model
    eval: bool = False                      # evaluate the model
    dataset: str = "./data/"                # directory that the dataset is located at
    # fraction of the train dataset split for testing
    test_size: float = 0.1
    model: str = "./models/model/"          # directory where the model file is stored
    tokenizer: str = "./tokenizer/"         # directory where the tokenizer is stored
    output: str = "./output/output.txt"     # output path for evaluation
    context_length: int = 1024              # size of the context window
    # directory where the pretrained model file is stored
    from_pretrained: str
    # number of batches at a time during training
    train_batch_size: int = 2
    eval_batch_size: int = 8                # number of batches at a time during eval
    # number of runs through the data during training
    epochs: int = 1
    save_steps: int = 5000                  # number of steps between saving the model
    eval_steps: int = 5000                  # number of steps between evaluation
    num_samples: int = 20                   # number of output samples

def main(config: Config):
    print("\nStarting in 5 seconds...")
    time.sleep(5)
    if config.train:
        data_files = [str(x) for x in Path(config.dataset).glob("**/*.txt")]

        if data_files == []:
            print(f"Error: {config.dataset} does not exist. Quitting.")
            sys.exit(1)

        if config.tokenizer == "":
            tokenizer = ByteLevelBPETokenizer()

            tokenizer.train(files=data_files, vocab_size=52_000, min_frequency=1, special_tokens=[
                "<s>",    # start of sequence
                "<pad>",  # pad sequence to correct length
                "</s>",   # end of sequence
                "<unk>",  # unknown token
                "<mask>",  # token to tell the model where to fill in
            ])

            if not os.path.exists(config.tokenizer):
                os.mkdir(config.tokenizer)

            tokenizer.save_model(config.tokenizer)

        dataset = load_dataset("text", data_files=data_files, split="train").shuffle(time.time_ns())

        tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer)
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        })

        def encode(element):
            outputs = tokenizer(
                element["text"],
                truncation=True,
                max_length=config.context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            return {"input_ids": outputs["input_ids"]}

        dataset = dataset.map(
            lambda x: {"text": "<s>" + x["text"] + "</s>"}, num_proc=os.cpu_count())
        dataset = dataset.map(encode, batched=True, remove_columns=[
            "text"], num_proc=os.cpu_count())
        dataset = dataset.train_test_split(test_size=config.test_size)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

        model: GPT2LMHeadModel

        if config.from_pretrained:
            model = GPT2LMHeadModel.from_pretrained(config.from_pretrained)
        else:
            model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=tokenizer.vocab_size,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token=tokenizer.eos_token_id,
                    n_positions=config.context_length,
                )
            )

        print(f"Model has {model.num_parameters():,} parameters")

        training_args = TrainingArguments(
            output_dir=config.model,
            overwrite_output_dir=True,
            logging_dir="logs",
            logging_strategy="steps",
            logging_steps=50,
            num_train_epochs=config.epochs,

            # adjust these depending on how much vram you have
            per_device_train_batch_size=config.train_batch_size,
            per_device_eval_batch_size=config.eval_batch_size,
            fp16=True,

            save_steps=config.save_steps,
            save_total_limit=5,
            prediction_loss_only=False,
            remove_unused_columns=False,

            evaluation_strategy="steps",
            eval_steps=config.eval_steps,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
        )

        trainer.train()
        trainer.save_model(config.model)

    if config.eval:
        tokenizer = GPT2TokenizerFast.from_pretrained(config.tokenizer)
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "pad_token": "<pad>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "mask_token": "<mask>",
        })

        model = GPT2LMHeadModel.from_pretrained(config.model).cuda()

        print(f"Model has {model.num_parameters():,} parameters")

        num_samples = config.num_samples

        print(f"Generating {num_samples} samples...")

        for sample in range(num_samples):
            inp = f"<s>"
            input_ids = tokenizer.encode(inp, return_tensors="pt").cuda()

            output = model.generate(
                input_ids,
                max_length=1024,
                # temperature=0.7,
                num_return_sequences=1,
                # top_k=50,  # refer to https://huggingface.co/blog/how-to-generate
                # top_p=1,
                do_sample=True,
                # repetition_penalty=1.1,

                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )[0]

            with open(config.output, "a+") as f:
                out = tokenizer.decode(output, skip_special_tokens=True)
                f.write(out + "\n")
                print(f"Done with sample {sample}/{config.num_samples}")


if __name__ == "__main__":
    main(Config())
