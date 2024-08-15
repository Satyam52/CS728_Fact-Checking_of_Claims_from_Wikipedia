# CS728-Fact-Checking of Claims from Wikipedia 

## Instruction to run the code
1. Run the command `bash main.sh` with appropriate parameters mentioned in the file
2. The model architecture is implemented in `model.py`
3. The training, evaluation, and loss function is implemented in `trainer.py`


## Proposed Solution

As already mentioned in the problem statement, we have to build a three-stage pipeline for this task specifically the stages being:
1. Top-50 Document Retrieval
2. Top-K Sentence Retrieval
4. Claim-Evidence NLI for Label Prediction

We implemented a comprehensive approach for evidence retrieval, incorporating dense indexing and Locality-Sensitive Hashing (LSH) techniques. Here's a summary of our methodology:

### Dense Indexing of Wikipedia Documents/Sentences

- We employed dense indexing to effectively index Wikipedia documents or sentences. This involved encoding the sentences using a BERT-based encoder, generating embeddings that were indexed for efficient retrieval.

### Encoding Wikipedia Sentences with MPNet

- Using an MPNet-based architecture, we encoded Wikipedia sentences to capture their semantic representations. These embeddings served as the foundation for our dense retrieval operations.

### Locality-Sensitive Hashing (LSH)

- LSH was applied to the indexed Wikipedia dump. This involved hashing the encoded claim embeddings and conducting similarity searches against the indexed Wikipedia embeddings.

The final pipeline that we used for this assignment is:
1. Top-50 Document Level Retrieval
2. Top-K Sentence Level Retrieval
3. NLI for Label Prediction

We describe these stages in further sections below.

### Stage 1: Document Retrieval

- We utilized the FAISS library to perform dense-indexing of the documents. We employed MPNET, a BERT-based sentence-transformers model that has a proven record of giving rich sentence embedding. We encoded each document of the wiki corpus and used FAISS to create a dense index. Then for each claim in train, dev, and test, we retrieved the top 50 documents from the Wikipedia corpus, which we use in stage 2.

### Stage 2: Sentence Retrieval

- Once we got access to the top 50 documents for each claim, we then used an MPNet-based architecture to get sentence embeddings for individual sentences and finally returned the Top 5 sentences for the downstream NLI task.
- For each claim in the train data, we have 3-4 evidence on average. For better training, we added 5 more hard negative samples.
- For hard sampling, we do inference in a zero-shot setting using our MPNet model for the sentences obtained from the stage 1 retrieval and get the top 5 sentences based on cosine similarity between claim embedding and evidence embedding. The irrelevant sentences in the top 5 ranks are considered as hard negative samples, which are used for penalizing the model for classifying them as negative. Then, we train the stage 2 retrieval system for ranking the sentences from evidence documents. So, having around a total of 7-8 evidence per instance makes the total dataset size around 800K. This led to a very high training time for the retrieval system. Specifically, a single epoch requires more than 15 hours of time. To implement this, we used MPNet-based encoders for claims and evidence sentences, which gave us embeddings to calculate cosine similarity and then sort the sentences based on this score. We also experimented with a few different architectures apart from MPNet, such as Mini-LM, Tiny BERT, and BERT. We adhered to MPNet due to its superior performance in later stages.

### Stage 3: Natural Language Inference

- We use all the gold-standard evidence corresponding to the given claim to train the NLI model. We explore two different setups for the natural language inference task.

#### Late Interaction

- This setup involves separately encoding the claim and evidences extracted for the given claim, and then the two embeddings are concatenated into a single vector. This concatenated vector is passed into a multilayer perceptron and projected into a 3-dimensional space. This embedding is normalized to obtain probability values using the softmax function.

#### Early Interaction

- In this setup, a concatenated string of claim and evidence is created and encoded together to get a single representation. For this, we concatenate the Claim and Evidences together by [SEP] token and encode the concatenated string with the model. We pass this representation to a multilayer perceptron to project it into a 3-dimensional space. We then use softmax to normalize the vectors into a probability distribution.

## Model Setup

For this assignment, we had to set up 3 different models for the three stages mentioned earlier. For document-level retrieval of stage 1, we used a zero-shot approach to retrieve and rank the documents from the FEVER dataset. A point to note is that the training dataset had 95K instances where the evidence information wasn’t available to use. We simply discarded those instances and performed all experiments on the remaining data points.

### Stage 1 Model Setup

- For document-level retrieval, we used the MPNet model which outputs a 768-dimensional embedding for each token with a maximum token length equal to 512 with truncation.
- Mean pooling across all tokens is performed to get a dense embedding for a document.
- The same model is used to encode the claim.
- LSH is implemented using FAISS with the 768-dimensional document embedding.

### Stage 2 Model Setup

- For the sentence-level retrieval task, we adhered to using Top-50 documents retrieved from Stage 1 retrieval.
- We used the MPNet encoder to get sentence embeddings for claims and evidences respectively.
- We didn’t concatenate all evidences together; instead, we individually extracted embeddings of each sentence and then took a similarity score of it with the claim sentence embeddings. However, we didn’t split the multi-sentence claims into individual sentences as it would have introduced additional overheads given the time constraints.
- Tiny-BERT, BERT-medium, MPNet, and Mini-LM were explored for the purpose of this task.

### Stage 3: NLI for Tagging

We consider the following hyperparameter settings for both setups:
- `train_batch_size` = 42
- `val_batch_size` = 16
- `epochs` = 3    
- `hidden_dim` = 768
- `learning_rate` = 1e-3   
- `Optimizer` = AdamW

We use MPNet as our base model for building the neural network. We specifically choose MPNet because of its better contextual representation capabilities since it relies on masked and permuted language modeling. This delivers the best of masked language modeling and permuted language modeling [1]. We use gold-standard evidences from the train set for fine-tuning the system and report the results on the labeled Dev set and unlabelled test set.

## Results & Analysis

### Retrieval

On Dev set:
- Hits @1: 0.0805
- Hits @3: 0.1910
- Hits @10: 0.2667
- Hits @100: 0.7097

- Recall @1: 0.0805
- Recall @5: 0.1859
- Recall @10: 0.2534
- Recall @100: 0.5135

### NLI for Tagging

#### Early Interaction
- Dev set: 95.7546
- Test set: 54.43

#### Late Interaction
- Dev set: 40.32
- Test set: 35.4 

We observe consistently better performance in early interaction setups. This is due to the all-to-all interaction between the claim and evidence tokens, which ensures better performance in terms of natural language inference. However, in the late interaction scenario, there are no token-level interactions between claim and evidence tokens. This might have led to the loss of important information that could have been utilized for comparing the claim and evidences for natural language inference.

## Insights and Conclusion

We note the important challenges of building a good retrieval system while implementing this assignment. As discussed in the earlier sections, building a good retrieval system involves a vast amount of data due to the nature of the task and, therefore, requires high computational resources and training time.

In the Natural Language Inference task, we observe that the late interaction-based setup performs poorly compared to the early interaction-based setup. The poor performance of late interaction compared to early interaction is mostly due to the inability of the multilayer perceptron to capture the relation between two different vector representations (one for claim and another for evidence).

The other possible reason behind the poor performance of both setups could be high noise due to the retrieval system. We use gold-standard evidence during the training for better learning; however, during the inference, the model can access evidence from the retrieval system, and therefore, the evidence may not always provide correct information for the judgment of the model. Hence, the retrieval system is a significant bottleneck for natural language inference systems in this three-staged pipeline. The primary issue is that it takes a significant amount of time and resources to train a retrieval system that provides good, high-quality evidence sentences.

