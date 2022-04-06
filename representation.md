# Representation

# Distributed representations 1: Word embedding

## Distributed Representations of Words and Phrases and their Compositionality

Before we advance to distributed representations of words, we first need to understand the "sparse representations"

### Understanding Sparse Representations

Examples:

1. One-hot encoding
2. Bag of words: without order information; representation: a length N vector, N is size of vocabulary. Can be represented as Binary Matrix, Count Matrix or TF-IDF Matrix. Can from bag of words to bag of grams.
    1. TF-IDF - can create word-document matrix
        
        ![Untitled](Representa%208c959/Untitled.png)
        
    2. PPMI - can create word-word matrix
    
    ![Untitled](Representa%208c959/Untitled%201.png)
    
    We add the alpha to the context word
    

Drawbacks: 

1. not captureing sematic correlations
2. vectors are sprse and high-dimensional

Source: DLT lecture notes by Chao Zhang, Georgia Tech

### Understanding Dimension Reduction and Topic Modeling

- Latent Semantic Analysis
    - Using SVD, mapping data into low-dimensional representation by only selecting top k topics
    - Source:  [https://www.youtube.com/playlist?list=PLroeQp1c-t3qwyrsq66tBxfR6iX6kSslt](https://www.youtube.com/playlist?list=PLroeQp1c-t3qwyrsq66tBxfR6iX6kSslt)

### Understanding [word embedding](https://arxiv.org/pdf/1301.3781.pdf)

- word2vec[co-occurrence statistics], Local context window methods
    - CBoW: use a window to predict center word
        
        ![Untitled](Representa%208c959/Untitled%202.png)
        
    - SkipGarm: use center word to predict surrounding words
        
        ![Untitled](Representa%208c959/Untitled%203.png)
        
        - Structure:
            - Two layer NN,two weight matrix. each time we will pass one center word, and each word need to be forward pass for k times.
            - the hidden layer doesn't use any activation function, which is directly passed to the output layer. The output layer using softmax probablility, get the word with the highest prob and compare to the  output word's on-hot encoding.
        - Objective: find word representations that are useful for predicting the surrounding words.
        - Output: the two weight matrices, and we can use one of them or average those two.
            - every word has two vectors, as central word and context word respectively, we can use any one of it or we combine those two. By default we will use the vector when the word in the center word

source: [https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c](https://towardsdatascience.com/skip-gram-nlp-context-words-prediction-algorithm-5bbf34f84e0c)

[https://www.youtube.com/watch?v=pOqz6KuvLV8](https://www.youtube.com/watch?v=pOqz6KuvLV8)

[https://arxiv.org/pdf/1301.3781.pdf](https://arxiv.org/pdf/1301.3781.pdf)

### About this Paper

This paper mainly discussed the extensions of Skip-gram model. First is to use hierarchical softmax to reduce computational complexity. Second is to use negative sampling to reduce noise. Third is to subsampling the frequent word like "a", "the".

source: [https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6](https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6)，[https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b](https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b)

- Hierarchical softmax
- Sub-Sampling
    - For each word we encounter in our training text, there is a chance that we will effectively delete it from the text. The probability that we cut the word is related to the word’s frequency. This is to delete some word with high frequency “the” “and”
- Negative Sampling
    - we are instead going to randomly select just a small number of “negative” words (let’s say 5) to update the weights for. (In this context, a “negative” word is one for which we want the network to output a 0 for). We will also still update the weights for our “positive” word (which is the word “quick” in our current example).

## [GloVe](https://nlp.stanford.edu/pubs/glove.pdf): Global Vectors for Word Representation

First this paper discussed the drawbacks of LSA and local context window methods:

- LSA: poorly on the word analogy task
- Local context window: poorly utilize statistics of corpus(such as global co-occurrence counts)
- And then it introduces the GloVe:
    - Basic idea:
        
        ![Untitled](Representa%208c959/Untitled%204.png)
        
        If the ratio is large, the probe word is related to *wᵢ* but not w*ⱼ,* if the ratio is small, then the prob word is related to **w*ⱼ* but not *wᵢ .* If the fraction is close to one, the prob word is close to the both words.
        
        To model this relationship:
        
        ![Untitled](Representa%208c959/Untitled%205.png)
        
        ![Untitled](Representa%208c959/Untitled%206.png)
        
        We can absorb log(*Xᵢ*) as a constant bias term since it is invariant of *k.*
        
        Therefore, **the dot product of two embedding matrices predicts the log co-occurrence count**.
        
    - Cost function: Mean Square Error to calculate the error in the ground truth and the predicted co-occurrence counts.
        
        ![Untitled](Representa%208c959/Untitled%207.png)
        
        When the co-occurrence count is higher or equal a threshold, say 100, the weight will be 1. Otherwise, the weight will be smaller, subject to the co-occurrence count. Here is the objective function in training the GloVe model.
        
    - Training: [https://towardsdatascience.com/a-comprehensive-python-implementation-of-glove-c94257c2813d](https://towardsdatascience.com/a-comprehensive-python-implementation-of-glove-c94257c2813d)
        
        ```python
        class GloVe(nn.Module):
        
            def __init__(self, vocab_size, embedding_size, x_max, alpha):
                super().__init__()
                self.weight = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_size,
                    sparse=True
                )
                self.weight_tilde = nn.Embedding(
                    num_embeddings=vocab_size,
                    embedding_dim=embedding_size,
                    sparse=True
                )
                self.bias = nn.Parameter(
                    torch.randn(
                        vocab_size,
                        dtype=torch.float,
                    )
                )
                self.bias_tilde = nn.Parameter(
                    torch.randn(
                        vocab_size,
                        dtype=torch.float,
                    )
                )
                self.weighting_func = lambda x: (x / x_max).float_power(alpha).clamp(0, 1)
            
            def forward(self, i, j, x):
                loss = torch.mul(self.weight(i), self.weight_tilde(j)).sum(dim=1)
                loss = (loss + self.bias[i] + self.bias_tilde[j] - x.log()).square()
                loss = torch.mul(self.weighting_func(x), loss).mean()
                return loss
        ```
        

> source: [https://www.youtube.com/watch?v=QoUYlxl1RGI](https://www.youtube.com/watch?v=QoUYlxl1RGI)
> 
> 
> [https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6](https://jonathan-hui.medium.com/nlp-word-embedding-glove-5e7f523999f6)
> 
> [https://blog.csdn.net/coderTC/article/details/73864097](https://blog.csdn.net/coderTC/article/details/73864097)
> 

# Distributed Representation 2: Deep Contextual Representation

## Deep contextualized word representations ([ELMo](https://arxiv.org/pdf/1802.05365.pdf))

- Two layer BiLSTM
    
    ![Untitled](Representa%208c959/Untitled%208.png)
    
- Advantage
    - Elmo provides the embedding of a word that is present inside a sentence
    - It allows the network to use morphological clues to form robust representations for out-of-vocabulary tokens unseen in training.
- Steps
    - The architecture above uses a character-level convolutional neural network (CNN) to represent words of a text string into raw word vectors
    - These raw word vectors act as inputs to the first layer of biLM
    - The forward pass contains information about a certain word and the context (other words) before that word
    - The backward pass contains information about the word and the context after it
    - This pair of information, from the forward and backward pass, forms the intermediate word vectors
    - These intermediate word vectors are fed into the next layer of biLM
    - The final representation (ELMo) is the **weighted sum of the raw word vectors and the 2 intermediate word vectors**

> [https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)
> 
> 
> [https://www.youtube.com/watch?v=YZerhaFMPTw&t=366s](https://www.youtube.com/watch?v=YZerhaFMPTw&t=366s)
> 

## GPT:

## BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

[https://www.youtube.com/watch?v=xI0HHN5XKDo](https://www.youtube.com/watch?v=xI0HHN5XKDo)

- Idea
    - **Masked language model**: Mask out k%(15%) input word then predict the masked words - (recent work: we can improve the 15%
    - **Next Sentence Prediction**: to predict if B is the next sentence of A or not

![Untitled](Representa%208c959/Untitled%209.png)

![Untitled](Representa%208c959/Untitled%2010.png)

![Untitled](Representa%208c959/Untitled%2011.png)

![Untitled](Representa%208c959/Untitled%2012.png)

![Untitled](Representa%208c959/Untitled%2013.png)

![Untitled](Representa%208c959/Untitled%2014.png)

![Untitled](Representa%208c959/Untitled%2015.png)