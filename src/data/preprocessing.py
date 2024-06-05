
from datasets import DatasetDict, Dataset
import random
from bert_score import BERTScorer

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration
)

class DialogSumDataset:
    def __init__(self, tokenizer, use_contrastive_loss=False, generate_qds=False, push_to_hf=False) -> None:
        self.tokenizer = tokenizer
        self.use_contrastive_loss = use_contrastive_loss
        self.generate_qds = generate_qds
        self.push_to_hf = push_to_hf

    def handle_data(self, data: DatasetDict) -> DatasetDict:
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokenized_dataset = data.map(self.preprocess_function, batched=True)
            tokenized_dataset = tokenized_dataset.remove_columns([key for key in data["train"][0].keys()])

            print("+++++++++++++++++++")
            print(tokenized_dataset)
            print("+++++++++++++++++++")
            
            return tokenized_dataset

        except Exception as e:
            print(f"Error while tokenizing data: {e}")
            raise e
        
    def preprocess_function(self, data: Dataset) -> Dataset:
        ## Create Query-Dialogue-Summary Instruction Dataset
        if self.generate_qds==True:
            scorer = BERTScorer(lang="en", rescale_with_baseline=True)
            
            inputs, targets = [], []
            
            checkpoint = "google/flan-t5-large"
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            model = T5ForConditionalGeneration.from_pretrained(checkpoint)

            for dialogue, summary in zip(data["dialogue"], data["summary"]):
                queries = self.generate_queries(model, tokenizer, dialogue, summary, 6)

                answerable_queries = []
                for query in queries:
                    ## Text based filtering
                    output = self.text_based_filtering(model, tokenizer, query, dialogue)
                    if "yes" in output.lower():
                        answerable_queries.append(query)

                n = len(answerable_queries)

                if n == 1:
                    inputs.append(f"###Instruction: {answerable_queries[0]} ###Input: {dialogue}. The generated summary should be around {len(summary)}")
                    targets.append(summary)

                if n > 1:
                    filtered_queries = []
                    scores = [[0.0]*n for _ in range(n)]

                    for i in range(n):
                        for j in range(n):
                            if i > j:
                                scores[i][j] = self.semantic_filtering(scorer, answerable_queries[i], answerable_queries[j])
                    
                    keep_indices = set(range(n))
                    for i in range(n):
                        for j in range(n):
                            if scores[i][j] > 0.7 and i > j:
                                keep_indices.discard(j)
                    
                    for i in sorted(keep_indices):
                        filtered_queries.append(answerable_queries[i])

                    for query in filtered_queries:
                        inputs.append(f"###Instruction: {query} ###Input: {dialogue}. The generated summary should be around {len(summary)}")
                        targets.append(summary)
                    
        
        if self.generate_qds==False:
            prefix = "Summarize the following conversation:\n\n###"
            suffix = "\n\nSummary: "
            inputs = [prefix + input + suffix for input in data["dialogue"]]
            targets = data["summary"]

        max_source_length = 1024
        max_target_length = 176

        data["input_ids"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        # data["attention_mask"] = self.tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True, return_tensors="pt").attention_mask
        data["labels"] = self.tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        
        # Generate negative examples:
        if self.use_contrastive_loss==True:
            negative_summaries = self.generate_negative_examples(data["summary"])
            data["negative_labels"] = self.tokenizer(negative_summaries, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
            print("\033[92mComplete generate negative examples!\033[00m")

        label_ignore_ids = []
        for label in data["labels"]:
            label_example = [l if l != 0 else -100 for l in label]
            label_ignore_ids.append(label_example)

        data["labels"] = label_ignore_ids

        return data
    
    def generate_negative_examples(self, summaries):
        negative_summaries = []
        for summary in summaries:
            words = summary.split()
            random.shuffle(words)
            negative_summaries.append(" ".join(words))
        return negative_summaries

    def generate_queries(self, model, tokenizer, summary, num_queries):
        input_text = "Generate an answerable and specific question based on the following context:. ###\nContext: " + summary
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=64, num_return_sequences=num_queries, do_sample=True)
        queries = [tokenizer.decode(outputs, skip_special_tokens=True) for output in outputs]
        return queries
    
    def text_based_filtering(self, model, tokenizer, query, summary):
        input_text = "Is the question fully answerable from the context without any guessing, yes or no?###\nQuestion: " + query + "###\nContext: " + summary + "###Answer: "
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        output_text = model.generate(input_ids, num_return_sequences=1)
        output = tokenizer.decode(output_text, skip_special_tokens=True)
        return output
    
    def semantic_filtering(self, scorer, query1, query2):
        score = scorer.score([query1], [query2])[0]
        return score


def preprocessing_data(data: DatasetDict, tokenizer, use_contrastive_loss=False, generate_qds=False, push_to_hf=False) -> DatasetDict:
    try:
        dataset_ds = DialogSumDataset(tokenizer, use_contrastive_loss, generate_qds, push_to_hf)
        tokenized_data = dataset_ds.handle_data(data)

        return tokenized_data

    except Exception as e:
        print(f"Error while pre-processing data: {e}")
        raise e