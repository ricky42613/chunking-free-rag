# chunking-free-rag
Implement paper: "Grounding Language Model with Chunking-Free In-Context Retrieval"
## Installation
- Create conda environment with `conda env create --file chunk-free.yaml`
## Usage
### Initialize a PrefixDecoder Object
```python=
prefix_decoder = PrefixDecoder()
```
-  Parameters
    - knowledge_file: File path to long article related to the question
        - default: `long_knowledge.txt`
    - model: Large language model. ex. llama
        - default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` 
    - tokenizer: Tokenizer for llm
        - default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
### Retrieve High Quality Knowledge
```python=
question = '.....'
high_quality_knowledge = prefix_decoder.retrieve_knowledge(question, top_k=2)
```
-  Parameters
    -  question: question input by user
    -  top_k: Retrieve passage in `top_k` highest prefix score
## Result (distance = 200)
### Test 1
> Question: What\'s the most significant news related to cybersecurity in this week?
> 
> Retrieved Passage: The US CISA was also affected by an Ivanti system vulnerability at the beginning of the year. Hackers targeted the US federal high-risk chemical critical infrastructure, infiltrating the Chemical Security Assessment Tool (CSAT) provided by CISA and successfully deploying a Web Shell. Another key infrastructure cybersecurity information tool, CISA Gateway, was also affected. The OWASP Foundation, well-known for publishing the top ten web application security risks, recently issued a data breach notification. Member resume files from 2006 to 2014 may have been leaked due to a configuration issue on an old Wiki web server.
### Test 2
> Question: What did Red Hat announce?
> 
> Red Hat also announced a related vulnerability, CVE-2024--3094, with several news articles related to this incident. Fortunately, this hidden backdoor was discovered early, and currently, only some Linux versions are affected, with the impact not being too widespread. However, this supply chain attack targeting open-source software is still under investigation and discussion. Not only the accidental discovery process but also the method of implanting the backdoor has surprised various sectors. The GitHub account involved in this incident had been created in 2021 and 2022, gradually gaining the trust of the original XZ maintainers. This infiltration process has attracted significant attention and has once again sparked discussions on the security of open-source software.
## Some Problem
- 論文中好像沒有說明sentence prefix是指句子的前幾個token，因此，這次實作以句子前3個字作為prefix
- 如果top-k sample decode選擇的解碼路徑都不是sentence prefix，最終就會沒有合法的selected prefix可以去生成high knowledge passage
   - 例如：sentence prefix有[[This, is, a], [I, have, the], [Here, are, some]]，如果在predict第一個token時，top-3 sample到的是[some, a, the]，那最終預測出的sentence prefix就沒辦法match到knowledge中的任何一個句子。
