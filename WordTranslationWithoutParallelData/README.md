## Overview
[Conneau, A., Lample, G., Ranzato, M. A., Denoyer, L., & Jégou, H. (2017). *Word Translation Without Parallel Data*.](https://arxiv.org/pdf/1710.04087.pdf)

As a first step towards fully unsupervised machine translation, this paper demonstrates how to build a bilingual dictionary between two languages without using any parallel corpora, by aligning monolingual word embedding spaces in a fully unsupervised way.

This work is built on [Mikolov, 2013] showing that it is possible exploit the similarities of monolingual embedding spaces to a mapping between monolingual embeddings. Using supervised approach, [Mikolov, 2013] successfully shows that a simple linear mapping *W* gives good result for this mapping. In addition, [Xing, 2015] show that such mapping can be improved by enforcing an orthogonality constraint on *W*.

Based on that starting point, the work described in the paper show that it is possible to learn and refine such a mapping with a fully unsupervized approach using adversarial learning. It is based on the following process (figure extracted from the paper):

![Overview](./img/overview.png)

- (A) As a starting point, we have two independently trained word embeddings (can be different domains or different languages). The goal is to train a mapping between these word embeddings. The mapping can be as simple as a linear mapping, but could be more complicated (any neural network).
- (B) A first version of this mapping is obtained using adversarial learning following [Goodfellow, 2014] - where the generator is the projection mapping, and the discriminator is trying to make a difference between source projected embedding and native target embedding
- (C) Based on this first mapping, a refinment method is proposed based on Procrustes solution [Schönemann, 1966]
- (D) With a new distance metric (CSLS) - dealing with *hubs* issue - source-target pairs are extracted

Key references:

* [Mikolov, T., Le, Q., & Sutskever, I. (2013) *Exploiting similarities among languages for machine translation*.](https://arxiv.org/abs/1309.4168)
* [Xing, C., Wang, D., Liu, C., & Lin, Y. (2015). *Normalized Word Embedding and Orthogonal Transform for Bilingual Word Translation*.](http://anthology.aclweb.org/N/N15/N15-1104.pdf)
* [Schönemann, P. H. (1966). *A generalized solution of the orthogonal Procrustes problem*](https://link.springer.com/article/10.1007/BF02289451)

Major follow-up:

* [Lample, G., Denoyer, L., & Ranzato, M. (2017) *Unsupervised Machine Translation Using Monolingual Corpora Only*.](https://arxiv.org/pdf/1711.00043.pdf)

## Step-by-Step analysis and Implementation notes

### The models
* The generator is a simple Linear mapping defined in `net.py`, `Generator` class. The class also implement `orthogonalityUpdate` method (section 3.3). This method is called after each gradient update. Any other model could be implemented - just think about adaptation of the `orthogonalityUpdate` method.
* The discriminator is a 2 layer network implemented also in `net.py` and as defined in 3.1. It has a single cell output activated by a sigmoid. The value is the probability of the input being a true target embedding.

```
Sequential (
  (0): Dropout (p = 0.1)
  (1): Linear (300 -> 2048)
  (2): LeakyReLU (0.01)
  (3): Linear (2048 -> 1)
  (4): Sigmoid ()
)
```
* Loss function is Binary Cross Entropy measure (`BCELoss`)

it would be possible to consider a two cell output - and use a softmax layer, but I think this approach is far simpler.

### Extracting source-target pairs
* Fast nearest neighbors is implemented using Facebook FAISS library [Johnson 2017] - using python binding 
* To calculate Cross-Domain Similarity Local Scaling (CSLS) - as defined in section 2.3 - the value \( r_S(y_t) \) is pre-calculated for the full target-dictionary. The results of FAISS `search` are rescored using the CSLS metrics
* !!note!! - I only implemented one-way CSLS. In 3.4, it is also mentioned that the mutual nearest neighbor are also considered to restrict the list of candidates.

### Evaluation
An internal EN-FR dictionary is provided and can be used for evaluation with the `--evalDict` option. Note: this is note the dictionary used by the authors but it gives same type of results.

The provided dictionary has multiple meanings for each simple word, so for the calculation of the precision, I just give credit if one hypothesis matches one of the available meaning.
Note that the dictionary does not contain inflected form, so the score is under-evaluated since the proposed meaning might be available but in lemmatized form. Also source words missing from the evaluation dictionary are not taken into account.

For instance - the first entry below is ignored since `men` is not part of the reference dictionary, and the second is considered as wrong because `peut` would normally count in P@5 but is in inflected form.

```
men hommes      eux     les     enfin   ,       alors   que     ainsi   même    pourtant
can éventuellement      eventuellement  faut    peut    sinon   exemple sachant ou      bien    effectivement
```

*It will be interesting to check the content of the authors dictionary for the evaluation*.

### Adversarial Training
For adversarial training, the process described in [Goodfellow, 2014] as been implemented:
* first, k-steps of discriminator update using *batchSize* projected source example, and *batchSize* native target. As suggested, we use label smoothing for calculating the loss of the native target.
* then, one mini-batch update of the generator is done propagating gradient from the discriminator with inverse loss function.

### Refinement
Refinement Procedure as described in 2.2 is not yet implemented

## Use the script

### Dependency

* `pytorch`
* `scipy`
* `progressbar`
* `FAISS` (more tricky to install)

GPU can be used if available with `--gpuid ID` option in the script. 

A luatorch version is also available - it includes the adversarial training, and the (non efficient) nearest neighbors extraction.

### Running it
* Get fasttext word embeddings
```
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.fr.vec
...
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.ko.vec
```

* Train a model:
```
python src/train.py wiki.en.vec  wiki.fr.vec --vocSize 50000 --gpuid 0 --nIter 40
```
note that the word embeddings will be saved to binary format so that next call can just load these binary files as following:
```
python src/train.py wiki.en.vec_50000.bin wiki.fr.vec_50000.bin --vocSize 50000  --gpuid 0 --nIter 40
```

Also - to speed-up calculation, the calculation of *rs(yt)* (formula 6) is done at the beginning of the script, and saved in file `wiki.fr.vec_200000.bin_rs_knn10` for further runs (depends only on vocabulary size and knn value).

Most of the parameters of the process can be set in commandline as follow:

```
WORD TRANSLATION WITHOUT PARALLEL DATA

positional arguments:
  srcemb                source word embedding
  tgtemb                target word embedding

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           initial random seed
  --vocSize VOCSIZE     vocabulary size
  --dim DIM             embedding size
  --hidden HIDDEN       discriminator hidden layer size
  --discDropout DISCDROPOUT
                        discriminator dropout
  --smoothing SMOOTHING
                        label smoothing value
  --samplingRange SAMPLINGRANGE
                        sampling range on vocabulary for adversarial training
  --beta BETA           orthogonality adjustment parameter
  --k K                 iteration of discriminator training for each iteration
  --batchSize BATCHSIZE
                        batch size
  --learningRate LEARNINGRATE
                        learning rate
  --decayRate DECAYRATE
                        decay rate
  --nEpochs NEPOCHS     number of epochs
  --knn KNN             number of neighbors to extract
  --distance DISTANCE   distance to use NN or CSLS
  --load LOAD           load parameters of generator
  --save SAVE           save parameters of generator
  --evalDict EVALDICT   dictionary for evaluation
  --gpuid GPUID
```

The 10k first entries of the dictionary with are dumped at the last epoch.

The option `--load file` and `--save file`, can be used to save and reload the generator state.

## Some results
* 10k first English-French entries are provided [here](./sample.md) - corresponding to a P@5 score of 62.04 (vs. P@1 77.8 in the paper without refinment but using the authors own dictionary). The hyper-parameters are the same than the ones in the paper (except batch size) 
* Unsupervized model selection: as shown in Figure 2 of the paper, the highest precision (evaluated on reference dictionary) is also corresponding to minimal average CSLS score over the first 10k entries - here at the epoch 59:
![Unsupervized Validation vs. Precision](./img/validation-precision.png)
* For the same run, the following graph shows the evolution of the generator loss, of the discriminator loss, and of the (unsupervized) validation score. It is interesting to see that the discrimator loss decreases continuously till epoch 43 then struggle to keep position while generator loss decreases. In this run, a continuous decay rate of 0.99 was used, and no halving of the learning rate done. Also - learning rates of generator and discriminator are the same. It could be interesting to investigate other strategies. 
![Losses evolution vs. Validation](./img/Losses-Validation.png)
* Finally, I tried for more challenging language pairs like Korean-English - the shape of the curves are really different. First the generator does not seem to fool *at all* the discriminator, and after few epoch of discriminator training, the loss of the generator training raise drastically. In parallel, the validation curve does not show any clear sign of minimum reaching. I do not have any reference Korean-English dictionary to get an evaluation of the results, but surfing through generated dictionary does not show any good alignment. In the paper, scores for more exotic languages like Russian and Chinese are also comparatively very low. Is the mapping for these languages possible at all?
![Same for Korean-English](./img/koen.png)


## Personal comments and Discussions
* Even without the refinment implementation, the results are as good as promised by the paper: without any bilingual knowledge, it is possible to build a relatively accurate word translation table. We could argue that since the word embeddings have been trained on wikipedia (see [fasttext](https://github.com/facebookresearch/fastText), there is some implicit aligned knowledge that is necessarily reflected in the embeddings - leading to these results. Still it is still a wonder that multilingual word embeddings can be aligned like this, and with such a simple transformation. It would be interesting to test with other embeddings built on different sources of data.

There are some limits in the extraction:

* The mapping can not really deal with polysemy, since all meanings for a source word will necessarily be in the same "neighborhood". This is also the very nature of these word embeddings forcing multiple meanings to share single representations. It could be interesting to see what would happen with adaptive skip-gram word vectors...
* Also, beyond polysemy and contextual mappings, what is also really missing is the notion multi-word expression which are critical for a human perspective.

If we think about the potential usage of this work, it is clearly not directly usable to build word translation table: for most language pairs, such resource is already available and without all the limitations faced with this approach, on the other hand quality for rarer language pairs is far lower and would be challenged very quickly by any traditional (human) resource building process.
However, the findings and implications of this work are a huge step forward better understanding of word embedding and knowledge representation, and this work is the _apetizer_ for the the main course - also published by facebook research team: *Unsupervised Machine Translation Using Monolingual Corpora Only*...

Keep tuned...
