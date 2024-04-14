import pysbd
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

class PrefixDecoder:
    def __init__(self, knowledge_file='long_knowledge.txt', model=None, tokenizer=None) -> None:
        self.segmentor = pysbd.Segmenter(language="en", clean=False)
        if tokenizer == None:
            self.tokenizer = LlamaTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        else:
            self.tokenizer = tokenizer
        if model == None:
            self.model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            self.model.eval()
        else:
            self.model = model
        with open(knowledge_file, 'r') as f:
            self.knowledge = f.read().replace('\\n', '\n')
            self.sents_in_knowledge = [s.strip() for s in self.segmentor.segment(self.knowledge)] 
            prompt_without_question = self._construct_prompt()
            inputs = tokenizer(prompt_without_question, return_tensors="pt")
            self.past_key_values = model.model(**inputs, return_dict=True).past_key_values
        self._build_prefix_trie()

    def _build_prefix_trie(self):
        self.all_prefix = set()
        self.prefix2sents = {}
        for idx, s in enumerate(self.sents_in_knowledge):
            ptr = self.prefix2sents
            for i in range(1,4):
                prefix_id = int(self.tokenizer(s, return_tensors="pt")['input_ids'][0][i])
                if prefix_id not in ptr:
                    ptr[prefix_id] = {'sents': []}
                ptr[prefix_id]['sents'].append(idx)
                ptr = ptr[prefix_id]
                self.all_prefix.add(prefix_id)
        

    def _construct_prompt(self):
        prompt = f'''<|system|>You are a friendly chatbot who always responds in the style of a pirate.</s><|user|>Below is an article, read the article and answer my question after the article. Now the article begins: {self.knowledge} Now the article ends. Select several sentences from the article to answer my question. Question: '''
        return prompt

    def predict_next_token(self, hidden_state):
        logits = self.model.lm_head(hidden_state)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, -1)
        return next_token_probs
    
    def get_hidden_state(self, question):
        prompt_postfix = f'{question}</s><|assistant|>'
        inputs = self.tokenizer(prompt_postfix, return_tensors="pt")
        ouputs = self.model.model(**inputs, past_key_values=self.past_key_values, return_dict=True)
        return ouputs.last_hidden_state

if __name__ == '__main__':
    prefix_decoder = PrefixDecoder()
    

   
