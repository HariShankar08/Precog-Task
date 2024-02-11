# Representations for Words, Phrases and Sentences - Programming Task

## Parts of the Task:

* ### Word Similarity (Constrained)
  
  **(Attempted):** Used multiple metrics, word embeddings to predict the similarity of two words as per the SimLex-999 dataset. (**Hill, Felix, Roi Reichart, and Anna Korhonen. "Simlex-999: Evaluating semantic models with (genuine) similarity estimation." *Computational Linguistics* 41.4 (2015): 665-695.**)

* ### Word Similarity (Unconstrained)
  
  **(Pending.)**

* ### Phrase Similarity (Constrained)
  
  **(Pending.)**

* ### Phrase Similarity (Unconstrained)
  
  **(Pending.)**

* ### Sentence Similarity (Constrained)
  
  **(Pending.)**

* ### Sentence Similarity (Unconstrained)
  
  **(Pending.)**
  
  ### 

## Word Similarity Task:

### Methodology -

Most content words (excluding stopwords) can be represented by a three tuple of values known as **Valence**, **Arousal** and **Dominance**, which aims to represent the word by the emotion evoked by its usage; in particular, its pleasantness (Valence), intensity (Arousal) and degree of control exerted (Dominance.) [1]

We refer to the work of Saif [2], where 20,000 English words were given real valued scores for each of the three dimensions by manual annotation using Best-Worst Scaling. [3] 

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

2. VAD Scores + Word2Vec Similarity

3. VAD Scores + Wordnet (Wu-Palmer) Similarity

4. VAD Scores + Wu-Palmer Similarity + Word2Vec Similarity

(TODO: Cite Wu-Palmer, Word2Vec)

### Observations -

| Features                 | Model | MSE                | <= 1SD | <= 2 SD |
| ------------------------ | ----- | ------------------ | ------ | ------- |
| VAD                      | SVM   | 5.53500159519488   | 138    | 182     |
| VAD                      | NN    | 5.339602947235107  | 57     | 126     |
| VAD + Word2Vec           | SVM   | 4.724532153923312  | 81     | 141     |
| VAD + Word2Vec           | NN    | 4.737236499786377  | 81     | 141     |
| VAD + WordNet            | SVM   | 4.255252351426032  | 83     | 144     |
| VAD + WordNet            | NN    | 4.5430169105529785 | 81     | 141     |
| VAD + Word2Vec + WordNet | SVM   | 3.6621559184695323 | 105    | 156     |
| VAD + Word2Vec + WordNet | NN    | 3.9122095108032227 | 105    | 151     |

### Analysis -

# Bibliography

1. C.E. Osgood, Suci G., and P. Tannenbaum. 1957. The measurement of meaning. University of Illinois Press.

2. Mohammad, Saif. "Obtaining reliable human ratings of valence, arousal, and dominance for 20,000 English words." *Proceedings of the 56th annual meeting of the association for computational linguistics (volume 1: Long papers)*. 2018.

3. Flynn, Terry N., and Anthony AJ Marley. *Best-worst scaling: theory and methods*. Diss. Edward Elgar, 2014.
