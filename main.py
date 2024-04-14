from prefix_decode import PrefixDecoder

if __name__ == '__main__':
    prefix_decoder = PrefixDecoder()
    while True:
        question = input("Input question: ")
        high_quality_knowledge = prefix_decoder.retrieve_knowledge(question, top_k=2) 
        print(high_quality_knowledge)