import torch
import time
from transformers import BertModel, BertTokenizer, BertConfig
from torch import nn


class BertEmbeddings:
    def __init__(self):
        # Print out all GPU and CPU devices
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        bert_dir_path = '/projappl/project_2006059/mixAEMeta/local_bert_uncased/'
        self.tokenizer = BertTokenizer.from_pretrained(bert_dir_path)
        self.Bert_model = BertModel.from_pretrained(bert_dir_path, output_hidden_states=True)

        self.Bert_model = self.Bert_model.to(self.device)

        if torch.cuda.device_count() > 1:
            #print("Using", torch.cuda.device_count(), "GPUs!")
            self.Bert_model = nn.DataParallel(self.Bert_model)
    def create_bert_emb(self, sentences, sys):
        if sys == 'BGL':
            max_length = 10
        if sys == 'TB':
            # max_length = 12 test
            max_length = 17
        if sys == 'SPIRIT':
            max_length = 16
        if sys == 'LIBERTY':
            max_length = 13
        # Set cache batch size depending on GPU memory
        cache_size = 10000

        # Start tracking time
        #start_time = time.time()
        # Set cache batch size depending on GPU memory
        embeddings = []

        self.Bert_model.eval()

        for i in range(0, len(sentences), cache_size):
            #print(f"Processing bert setence batch starting at index {i}", flush=True)
            batch_sentences = sentences[i: i + cache_size]
            # Tokenize sentences in current batch
            tokenized_batches = self.tokenizer(batch_sentences, truncation=True, padding='max_length',
                                               add_special_tokens=True,
                                               return_tensors='pt', max_length=max_length)

            # Move tokenized input of current batch to GPU
            tokens_tensor = tokenized_batches['input_ids'].to(self.device, non_blocking=True)
            attention_mask = tokenized_batches['attention_mask'].to(self.device, non_blocking=True)

            with torch.no_grad():
                # print(f"Getting outputs for batch starting at index {i}", flush=True)
                outputs = self.Bert_model(tokens_tensor, attention_mask=attention_mask)

            hidden_states = outputs[2]
            token_vecs = hidden_states[-1]
            sentence_embs = torch.mean(token_vecs, dim=1)

            embeddings.append(sentence_embs)
            torch.cuda.empty_cache()
        embeddings = torch.cat(embeddings, dim=0)

        return embeddings.cpu()
