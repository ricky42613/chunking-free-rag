import pysbd
from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import heapq 
import math
from tqdm import tqdm

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
            self.prompt_without_question = self._construct_prompt()
            self._precompute()
        self._build_prefix_trie()

    def _precompute(self):
        self.inputs = self.tokenizer(self.prompt_without_question, return_tensors="pt")
        outputs = self.model.model(**self.inputs, return_dict=True)
        self.knowledge_logits = self.model.lm_head(outputs.last_hidden_state)
        self.past_key_values = outputs.past_key_values

    def _build_prefix_trie(self):
        self.id2token = {v:k for k,v in self.tokenizer.get_vocab().items()}
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
        prompt = f'''Below is an article, read the article and answer my question after the article. Now the article begins: {self.knowledge} Now the article ends. Select several sentences from the article to answer my question. Question: '''
        return prompt
    
    def _prefix_match_sents(self, toks):
        ptr = self.prefix2sents
        for t in toks:
            if t not in ptr:
                return []
            ptr = ptr[t]
        return ptr['sents']

    def _predict_next_token(self, hidden_state):
        logits = self.model.lm_head(hidden_state)
        logits = logits.float()
        next_token_logits = logits[:, -1, :]
        next_token_probs = torch.softmax(next_token_logits, -1)
        return next_token_probs

    def _get_combined_embedding(self, question):
        input_tokens = self.tokenizer(question, return_tensors="pt").input_ids[0]
        past_key_values = self.past_key_values
        for i in range(1, len(input_tokens)):
            current_inputs = {}
            current_inputs['input_ids'] = torch.tensor([input_tokens[i]])[:, None]
            current_inputs['attention_mask'] = torch.tensor([1])[:, None]
            ouputs = self.model.model(**current_inputs, past_key_values=past_key_values, return_dict=True)
            past_key_values = ouputs.past_key_values
        return ouputs

    def _get_topk_prefixes(self, current_status, top_k = 2):
        ret = []
        status = current_status
        for _ in tqdm(range(5)):
            next_status = []
            for output, score, prev_toks in status:
                last_hidden_state = output.last_hidden_state
                next_token_probs = self._predict_next_token(last_hidden_state)
                top_k_candidates = []
                for id in self.all_prefix:
                    if len(top_k_candidates) >= top_k:
                        if top_k_candidates[0][0] < next_token_probs[0][id]:
                            heapq.heappop(top_k_candidates)
                            heapq.heappush(top_k_candidates, (next_token_probs[0][id], id))
                    else:
                        heapq.heappush(top_k_candidates, (next_token_probs[0][id], id))
                for prob, prefix_id in top_k_candidates:
                    next_input_ids = torch.tensor([prefix_id])[:, None]
                    next_attn_mask =  torch.tensor([1])[:, None]
                    new_input = {'input_ids': next_input_ids, 'attention_mask': next_attn_mask}
                    past_key_values = output.past_key_values
                    new_ouput = self.model.model(**new_input, past_key_values=past_key_values, return_dict=True)
                    new_prev_toks = prev_toks + [prefix_id]
                    new_score = score+math.log(prob)
                    match_prefix = self._prefix_match_sents(new_prev_toks)
                    if len(match_prefix) == 1:
                        ret.append({'toks': new_prev_toks, 'score': new_score/len(new_prev_toks), 'idx': match_prefix[0]})
                    if len(match_prefix) > 1:
                        next_status.append((new_ouput, new_score, new_prev_toks))
                status = next_status
        ret.sort(key=lambda item: item['score'], reverse=True)
        return ret

    def find_beg_of_seq(self, promptToks, prefixToks):
        prefix_len = len(prefixToks)
        prefix_str = ''.join([self.id2token[id_].replace('▁', ' ') for id_ in prefixToks])
        for i in range(len(promptToks)):
            tmp = ''.join([self.id2token[int(id_)].replace('▁', ' ') for id_ in promptToks[i:i+prefix_len]])
            if prefix_str.strip() == tmp.strip():
                return i
        return -1

    def find_end_of_seq(self, beg_idx, logits, end_of_sent='</s>', distance=200):
        len_of_prompt = len(self.inputs.input_ids[0])
        end_of_sent_id = self.tokenizer.get_vocab()[end_of_sent]
        max_score = float('-inf')
        max_idx = -1
        for i in range(beg_idx, min(beg_idx+distance, len_of_prompt)):
            end_score = float(logits[0][i][end_of_sent_id])
            if max_score < end_score:
                max_score = end_score
                max_idx = i
        return max_idx


    def retrieve_knowledge(self, question):
        ret = []
        outputs = self._get_combined_embedding(question)
        selected_prefixes = self._get_topk_prefixes([(outputs, 0, [])])
        prompt_toks = [int(i) for i in self.inputs.input_ids[0]]
        for cand in selected_prefixes:
            sent_beg_idx = self.find_beg_of_seq(prompt_toks, cand['toks'])
            end_idx = self.find_end_of_seq(sent_beg_idx, self.knowledge_logits)
            high_quality_span = self.inputs.input_ids[0][sent_beg_idx: end_idx]
            high_quality_text = self.tokenizer.batch_decode([high_quality_span], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            ret.append(high_quality_text)
        return ret

if __name__ == '__main__':
    prefix_decoder = PrefixDecoder()
    

   
