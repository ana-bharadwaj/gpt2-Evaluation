from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, EvalPrediction
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import numpy as np
from scipy.special import softmax
from sklearn.metrics import log_loss


def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    labels = p.label_ids
    probabilities = softmax(logits, axis=-1)
    loss = log_loss(labels.flatten(), probabilities.reshape(-1, probabilities.shape[-1]),
                    labels=[i for i in range(logits.shape[-1])])
    perplexity = np.exp(loss)
    return {"perplexity": perplexity}


def fine_tune_gpt2(model_name, train_file, validation_file, output_dir):
    # Load GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load training dataset
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128)

    # Load validation dataset
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=validation_file,
        block_size=128)

    # Create data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_strategy='epoch',
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        save_total_limit=2,
    )

    # Train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Fine-tune the model
fine_tune_gpt2("gpt2", "/home/ana/PycharmProjects/peopleInstii/intents/train_data.txt", "/home/ana/PycharmProjects/peopleInstii/intents/validation_data.txt", "output")
