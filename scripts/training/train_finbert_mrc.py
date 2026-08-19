import os
import json
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset

def load_squad_data(file_path):
    with open(file_path, 'r') as f:
        squad_dict = json.load(f)
        
    contexts = []
    questions = []
    answers = []
    
    for group in squad_dict['data']:
        for paragraph in group['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # Handles multiple answers, taking the first one
                if 'answers' in qa and len(qa['answers']) > 0:
                    ans_obj = qa['answers'][0]
                    answer_text = ans_obj['text']
                    answer_start = ans_obj['answer_start']
                    
                    contexts.append(context)
                    questions.append(question)
                    answers.append({'text': [answer_text], 'answer_start': [answer_start]})
                    
    return Dataset.from_dict({
        'context': contexts,
        'question': questions,
        'answers': answers
    })

def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def train_mrc(data_path, output_dir, model_name="ahmedrachid/FinancialBERT", epochs=3, batch_size=4, smoke_test=False):
    print(f"Loading data from {data_path}...")
    dataset = load_squad_data(data_path)
    
    # Split
    dataset = dataset.train_test_split(test_size=0.1)
    
    print(f"Loading Tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    tokenized_datasets = dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    print(f"Loading Model: {model_name}")
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    
    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs if not smoke_test else 1,
        weight_decay=0.01,
        save_total_limit=2,
        max_steps=10 if smoke_test else -1, # Limit steps for smoke test
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=DefaultDataCollator(),
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/finbert_mrc_train_synthetic.json")
    parser.add_argument("--output_dir", type=str, default="models/finbert_mrc_custom")
    parser.add_argument("--model_name", type=str, default="ahmedrachid/FinancialBERT") # Using base FinancialBERT as pre-trained starter
    # Or user requested "bert-base-financial", assuming existing specific model.
    # ahmedrachid/FinancialBERT-SQuAD2 is already finetuned - we can continue finetuning or start fresh.
    # Starting fresh from FinancialBERT is more rigorous "Domain Adaptation".
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--smoke_test", action="store_true")
    
    args = parser.parse_args()
    
    # Resolve absolute path for Default
    if not os.path.isabs(args.data_path):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        args.data_path = os.path.join(BASE_DIR, args.data_path)
        
    train_mrc(args.data_path, args.output_dir, args.model_name, args.epochs, smoke_test=args.smoke_test)
