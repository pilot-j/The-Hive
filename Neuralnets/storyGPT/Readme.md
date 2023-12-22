# TODO.md


### Todo

- [x] Add executable notebook file
- [ ] Implement FlastAttention in place of vanilla attention
- [ ] Implement changes to tranformer architecture as introduced in LLama 1
- [x] Training on full dataset 
- [ ] Add python files that can be executed on system (under progress)
- [ ] Create web app that can take user prompt and generate short story using StreamLit
    - [x] Create interface
    - [ ] Integrate model (under progress)
  

# GPT-Inspired Story Generator

![image](https://github.com/pilot-j/The-Hive/assets/120032562/7b5b9d65-ecb0-44d7-bbd6-1388a1328afb)
## Introduction

This project is an attempt into building a generative pre-trained language model inspired by the principles of GPT (Generative Pre-trained Transformer). This is an autoregressive model capable of generating coherent English text. Here, I take the opportunity to reproduce(and test) the claims of this interesting research paper ["TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"](https://arxiv.org/pdf/2305.07759.pdf).
The current model can produce coherent english sentences, though mostly not related to each other (since the context window is only 512 tokens!!)
Update:
* I came across this interesting paper(currently under open review ) [In the WildChat](https://openreview.net/pdf?id=Bl8u7ZRlbM) in which the authors have created a dataset of ChatGpt interactions with its users. To test its utility in finetuning the model's story writing capacity I am now training on a small english subset of the dataset (attempt failed. Model context window has to be increased to successfully generate coherent stories.)
* To reduce the training time and generate better output for prompts I have moved from character level (naive!) tokenisation to custom subword level tokeniser. I have used google's [SentencePiece](https://github.com/google/sentencepiece) (unigram algorithm) to train a tokeniser with vocabulary size of 8K tokens.
* I tried to create a custom "coxtext aware" tokeniser by implementing word2vec model for encoding. However, it was difficult to create an excatly reversible vec2word model for the decoding scheme of the tokeniser as a result of which I didn't use this method. Irreversibilty in tokenisation is a problem adequately addressed by sentencepiece however, the tokenisation is based on efficiecny and not neccessarily on context relation between sub words.

The initial model is tested agasint a subset of the original TinyStories dataset. The TinyStories dataset is designed to investigate the minimal requirements for language models to produce coherent and meaningful English text.

## Acknowledgments

This project draws inspiration from [Andrej Karpathy's nanoGPT](https://youtu.be/kCc8FmEb1nY?feature=shared) and the informative blog post on GPT-2 by [Jay Alammar](https://jalammar.github.io/illustrated-gpt2/). The guidance provided in these resources has been instrumental in shaping the direction of this language model.

---

**Note:** This README is a work in progress and will be refined further as the project evolves. Contributions, feedback, and collaboration are welcome!


