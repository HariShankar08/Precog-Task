# Representations for Words, Phrases and Sentences - Programming Task

## Parts of the Task:

* ### Word Similarity (Constrained)
  
  **(Attempted):** Used multiple metrics, word embeddings to predict the similarity of two words as per the SimLex-999 dataset. [1]

* ### Word Similarity (Unconstrained)
  
  **(Attempted):** Used previously used metrics in constrained version, as well as BERT similarity of word embeddings to predict the similarity as in the same SimLex-999 dataset. 

* ### Phrase Similarity (Constrained)
  
  **(Attempted):** Predicting whether two phrases are similar (i.e can be used interchangeably) in the context of a particular sentence. [2]

* ### Phrase Similarity (Unconstrained)
  
  **(Attempted):** Used BERT embeddings and similarity to predict whether the two phrases can be used interchangeably in a particular context. 

* ### Sentence Similarity (Constrained)
  
  **(Pending.)** [3]

* ### Sentence Similarity (Unconstrained)
  
  **(Pending.)**
  
  ### 

## Word Similarity Task:

### Methodology -

Most content words (excluding stopwords) can be represented by a three tuple of values known as **Valence**, **Arousal** and **Dominance**, which aims to represent the word by the emotion evoked by its usage; in particular, its pleasantness (Valence), intensity (Arousal) and degree of control exerted (Dominance.)  [4]

We refer to the work of Saif [5], where 20,000 English words were given real valued scores for each of the three dimensions by manual annotation using Best-Worst Scaling. [6] 

The aim of the task was to predict the SimLex score provided in the data, using different word representations and embeddings.

Various ML models were tested for this case (Linear Regression, SVM, Random Forest) and it was determined that SVMs were best suited for the task; which was decided by comparing mean cross-validation scores on 5 splits of the data.

We compared the performance of SVM and Neural Networks for the various embedding approaches followed. The following represents the general architecture of the Neural Networks trained;

| Layer Type | Number of Nodes | Activation         |
| ---------- | --------------- | ------------------ |
| Dense      | 128             | ReLU (Input Layer) |
| Dense      | 64              | ReLU               |
| Dense      | 1               | None (Output)      |

The Neural models were all trained on 100 epochs, with a batch size of 16. The Adam Optimizer was used during training.

The SVM and Neural Model were compared after incrementally adding new features on top of the previous model. The following features were used:

1. VAD Scores

2. VAD Scores + Word2Vec Similarity [8]

3. VAD Scores + Wordnet (Wu-Palmer) Similarity [9]

4. VAD Scores + Wu-Palmer Similarity + Word2Vec Similarity

(TODO: Cite Wu-Palmer, Word2Vec)

### Observations -

| Features                        | Model | MSE                | <= 1SD | <= 2 SD |
| ------------------------------- | ----- | ------------------ | ------ | ------- |
| VAD                             | SVM   | 5.264556766795256  | 78     | 134     |
| VAD                             | NN    | 5.546095848083496  | 77     | 128     |
| VAD + Word2Vec                  | SVM   | 5.332867766938792  | 80     | 130     |
| VAD + Word2Vec                  | NN    | 5.1631999015808105 | 77     | 131     |
| VAD + WordNet                   | SVM   | 4.078742705415954  | 89     | 151     |
| VAD + WordNet                   | NN    | 4.390047550201416  | 86     | 147     |
| VAD + Word2Vec + WordNet        | SVM   | 4.0478806839649435 | 91     | 152     |
| VAD + Word2Vec + WordNet        | NN    | 4.52810525894165   | 90     | 140     |
| VAD + BERT Similarity + WordNet | SVM   | 4.0478806839649435 | 91     | 152     |
| VAD + BERT Similarity + WordNet | NN    | 4.312497138977051  | 87     | 149     |

### Analysis -

Overall, we see that Neural Networks are underperforming the SVM counterparts. This may be as a result of the small train/validation size for the dataset provided for this task. A surprising result that is noted is the same performance of the model using Word2Vec similarity as opposed to similarity of BERT embeddings.

It is believed that this is in part because the fact that BERT embeddings are contextual, and so part of the information encoded in its embedding is already latent in our dataset by virtue of the VAD scores and WordNet similarity. In contrast, the WordNet similarity only depends on co-occurences in the corpus trained, and such does not encode meaning similar to the VAD scores/Wu-Palmer similarity.

The model trained on VAD Scores, Word2Vec Similarity and Wu-Palmer Similarity performs well in distinguishing between words that are commonly antonyms (such as `accept` and `forgive`; or `make` and `destroy`), often predicting a score within 2 standard deviations from the value provided in the dataset. (With exceptions; `floor` and `ceiling` received high similarity scores from the model.)

The model's performance for synonym pairs varies with the specific pair used. For example, `boundary` and `border` received a significantly low similarity. On the other hand, `know` and `comprehend`, `frustration` and `anger` received better similarity scores within 2 standard deviations of the value provided in the dataset.

  



## Phrase Similarity Task:

### Methodology -

In an isolated scenario, devoid of external context from a sentence, it was assumed that phrases can be compared by averaging the individual word similarity scores using a greedy matching approach, similar to that used in the paper by Zhang et al. [7]

However, as mentioned above, this metric is devoid of context, and is only so useful in a task checking the similarity of a phrase in a particular target sentence. 

We have used a metric based on the average similarity of the content words in the sentence and the root of the phrase using the WordNet (Wu-Palmer) similarity metric. This metric is based on the assumptions that lead to the Lesk Algorithm [10] for Word Sense Ambiguation, that the words in the surrounding context would be similar to a particular sense of the word.

### Observations -

We observe the following performance metrics among various models:

| Features                                          | Model                          | Accuracy |
| ------------------------------------------------- | ------------------------------ | -------- |
| Greedy Matching + Overlap                         | Logistic Regression            | 50.35%   |
| Greedy Matching + Overlap                         | Random Forest (100 Estimators) | 43.79%   |
| Greedy Matching + Overlap                         | NN                             | 50.05%   |
| Greedy Matching + Overlap                         | SVM                            | 53.64%   |
| Greedy Matching + Overlap (Sentence Transformers) | Logistic Regression            | 52.81%   |
| Greedy Matching + Overlap (Sentence Transformers) | Random Forest (100 Estimators) | 50.90%   |
| Greedy Matching + Overlap (Sentence Transformers) | NN                             | 54.01%   |
| Greedy Matching + Overlap (Sentence Transformers) | SVM                            | 50%      |

The Neural Model used for this task is described below:

| Layer | Number of Nodes | Activation       |
| ----- | --------------- | ---------------- |
| Dense | 128             | ReLU (Input)     |
| Dense | 64              | ReLU             |
| Dense | 1               | Sigmoid (Output) |

* Optimizer: Adam

* Epochs: 100

* Batch Size: 64



Accuracy **is** rather low. We believe that this may be improved with other Neural model architectures, and also using contextual text embedding approaches such as BERT or ELMo.

### Analysis -



## Sentence Similarity





# Bibliography



1. Hill, Felix, Roi Reichart, and Anna Korhonen. "Simlex-999: Evaluating semantic models with (genuine) similarity estimation." *Computational Linguistics* 41.4 (2015): 665-695

2. Pham, Thang M., et al. "PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search." *arXiv preprint arXiv:2207.09068* (2022).

3. Zhang, Yuan, Jason Baldridge, and Luheng He. "PAWS: Paraphrase adversaries from word scrambling." *arXiv preprint arXiv:1904.01130* (2019).

4. C.E. Osgood, Suci G., and P. Tannenbaum. 1957. The measurement of meaning. University of Illinois Press.

5. Mohammad, Saif. "Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words." *Proceedings of the 56th annual meeting of the association for computational linguistics (volume 1: Long papers)*. 2018.

6. Flynn, Terry N., and Anthony AJ Marley. *Best-worst scaling: theory and methods*. Diss. Edward Elgar, 2014.

7. Zhang, Tianyi, et al. "Bertscore: Evaluating text generation with bert." *arXiv preprint arXiv:1904.09675* (2019).

8. Church, Kenneth Ward. "Word2Vec." *Natural Language Engineering* 23.1 (2017): 155-162.

9. Wu, Zhibiao, and Martha Palmer. "Verb semantics and lexical selection." *arXiv preprint cmp-lg/9406033* (1994).

10. Lesk, Michael. "Automatic sense disambiguation using machine readable dictionaries: how to tell a pine cone from an ice cream cone." *Proceedings of the 5th annual international conference on Systems documentation*. 1986.
