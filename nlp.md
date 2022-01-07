================================================================================
RoBERTa: A Robustly Optimized BERT Pretraining Approach
https://arxiv.org/abs/1907.11692
================================================================================

(1) We present a set of important BERT design choices and training strategies and introduce alternatives that lead to better downstream task performance; 
(2) We use a novel dataset, CCNEWS, and confirm that using more data for pretraining further improves performance on downstream tasks; 
(3) Our training improvements show that masked language model pretraining, under the right design choices, is competitive with all other recently published methods. 

*** Great summary of BERT ***

Training Setup:
BERT is optimized with Adam (Kingma and Ba, 2015) using the following parameters: Î˛1 = 0.9, Î˛2 = 0.999, ÇŤ = 1e-6 and L2 weight decay of 0.01. 
The learning rate is warmed up over the first 10,000 steps to a peak value of 1e-4, and then linearly decayed. 
BERT trains with a dropout of 0.1 on all layers and attention weights, and a GELU activation function (Hendrycks and Gimpel, 2016). 
BERT models are pretrained for S = 1,000,000 updates, with minibatches containing B = 256 sequences of maximum length T = 512 tokens.

RoBERTA follows BERT setup except for the peak learning rate and number of warmup steps, which are tuned separately for each setting. 
We additionally found training to be very sensitive to the Adam epsilon term.
We found setting Î˛2 = 0.98 to improve stability when training with large batch sizes.
ROBERTA is pretrained with sequences of at most T = 512 tokens. 
We train only with full-length sequences.
We train with mixed precision floating point arithmetic on DGX-1 machines, each with 8 x 32GB Nvidia V100 GPUs interconnected by Infiniband.

RoBERTa configuration uses:
dynamic masking (Section 4.1), 
FULL-SENTENCES without NSP loss (Section 4.2), 
large mini-batches (Section 4.3) - 2k and 8k
larger byte-level BPE (Section 4.4).


================================================================================
ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
https://arxiv.org/abs/1909.11942
================================================================================

ALBERT-large has about 18x fewer parameters compared to BERT-large, 18M versus 334M. 

The backbone of the ALBERT architecture is similar to BERT in that it uses a transformer encoder (Vaswani et al., 2017) with GELU nonlinearities (Hendrycks & Gimpel, 2016). 
Following Devlin et al. (2019), we set the feed-forward/filter size to be 4H and the number of attention heads to be H/64.

There are three main contributions that ALBERT makes over the design choices of BERT.

Factorized embedding parameterization:
In BERT, as well as subsequent modeling improvements such as XLNet (Yang et al., 2019) and RoBERTa (Liu et al., 2019), the WordPiece embedding size E is tied with the hidden layer size H, i.e., E == H.
From a practical perspective, natural language processing usually require the vocabulary size V to be large. 
If E == H, then increasing H increases the size of the embedding matrix, which has size V x E. 
This can easily result in a model with billions of parameters, most of which are only updated sparsely during training.
Instead of projecting the one-hot vectors directly into the hidden space of size H, ALBERT first projects them into a lower dimensional embedding space of size E, and then project it to the hidden space. 
By using this decomposition, we reduce the embedding parameters from O(V x H) to O(V x E + E x H).

Cross-layer parameter sharing:
The default decision for ALBERT is to share all parameters across layers. 
Results show that weight-sharing has an effect on stabilizing network parameters. 

Inter-sentence coherence loss:
In addition to the masked language modeling (MLM) loss (Devlin et al., 2019), BERT uses an additional loss called next-sentence prediction (NSP). 
NSP is a binary classification loss for predicting whether two segments appear consecutively in the original text, as follows: positive examples are created by taking consecutive segments from the training corpus; negative examples are created by pairing segments from different documents; positive and negative examples are sampled with equal probability. 
However, subsequent studies (Yang et al., 2019; Liu et al., 2019) found NSPâs impact unreliable and decided to eliminate it. 
We conjecture that the main reason behind NSPâs ineffectiveness is its lack of difficulty as a task, as compared to MLM. 
ALBERT proposes a loss based primarily on coherence. 
It uses a sentence-order prediction (SOP) loss.
The SOP loss uses as positive examples the same technique as BERT (two consecutive segments from the same document), and as negative examples the same two consecutive segments but with their order swapped. This forces the model to learn finer-grained distinctions about discourse-level coherence properties. 

================================================================================
DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
https://arxiv.org/abs/1910.01108
================================================================================

n this paper, we show that it is possible to reach similar performances on many downstream-tasks using much smaller language models pre-trained with knowledge distillation.
We also show that our compressed models are small enough to run on the edge, e.g. on mobile devices.

Student architecture:
In the present work, the student - DistilBERT - has the same general architecture as BERT. 
The token-type embeddings and the pooler are removed while the number of layers is reduced by a factor of 2. 

Student initialization:
we initialize the student from the teacher by taking one layer out of two.

Distillation:
We applied best practices for training BERT model recently proposed in Liu et al. [2019].
DistilBERT is distilled on very large batches leveraging gradient accumulation (up to 4K examples per batch) using dynamic masking and without the next sentence prediction objective.

Data and compute power:
We train DistilBERT on the same corpus as the original BERT model: a concatenation of English Wikipedia and Toronto Book Corpus [Zhu et al., 2015]. 
DistilBERT was trained on 8 16GB V100 GPUs for approximately 90 hours. 
For the sake of comparison, the RoBERTa model [Liu et al., 2019] required 1 day of training on 1024 32GB V100.

Training loss:
The student is trained with a distillation loss over the soft target probabilities of the teacher: 
$L_{ce} = \sum_i t_i * log(s_i)$
where $t_i$ (resp. $s_i$) is a probability estimated by the teacher (resp. the student). 
This objective results in a rich training signal by leveraging the full teacher distribution. 
Following Hinton et al. [2015] we used a softmax-temperature: 
$\sum_i \frac{exp(z_i/T)}{\sum_j exp(z_j/T)}$
where $T$ controls the smoothness of the output distribution and z_i is the model score for the class i.
The same temperature T is applied to the student and the teacher at training time. 
At inference, T is set to 1 to recover a standard softmax.

Training objective:
Is a linear combination of the distillation loss Lce with the supervised training loss, in our case the masked language modeling loss Lmlm [Devlin et al., 2018]. 
We found it beneficial to add a cosine embedding loss (Lcos) which will tend to align the directions of the student and teacher hidden states vectors.

================================================================================
ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators
https://arxiv.org/abs/2003.10555
================================================================================

While more effective than conventional language-model pre-training due to learning bidirectional representations, these masked language modeling (MLM) approaches incur a substantial compute cost because the network only learns from 15% of the tokens per example.
As an alternative, we propose replaced token detection, a pre-training task in which the model learns to distinguish real input tokens from plausible but synthetically generated replacements. 
The method corrupts the input by replacing some tokens with samples from a proposal distribution, which is typically the output of a small masked language model. 
We then pre-train the network as a discriminator that predicts for every token whether it is an original or a replacement.
 Incontrast, MLM trains the network as a generator that predicts the original identities of the corrupted tokens. 

Although the approach is reminiscent of training the discriminator of a GAN, our method is not adversarial in that the generator producing corrupted tokens is trained with maximum likelihood due to the difficulty of applying GANs to text (Caccia et al., 2018).

ELECTRA substantially outperforms MLM-based methods such as BERT and XLNet given the same model size, data, and compute (see Figure 1). 
For example, we build an ELECTRA-Small model that can be trained on 1 GPU in 4 days, and that outperforms a comparably small BERT model by 5 points on GLUE.
Our approach also works well at large scale, where we train an ELECTRA-Large model that performs comparably to RoBERTa (Liu et al., 2019) and XLNet (Yang et al., 2019), despite having fewer parameters and using 1/4 of the compute for training. 
Training ELECTRA-Large further results in an even stronger model that outperforms ALBERT (Lan et al., 2019) on GLUE.
It has 1/20th the parameters and requires 1/135th the pre-training compute of BERT-Large.

Our approach trains two neural networks, a generator G and a discriminator D. 
Each one primarily consists of an encoder (e.g., a Transformer network) that maps a sequence on input tokens x = [x1, ..., xn] into a sequence of contextualized vector representations h(x) = [h1, ..., hn]. 

G (generator) is a MLM (mask language model).
$p_G(z_t|x) = exp(e(x_t)^T h_G(x)_t) / sum_x^prime exp(e(x_t)^T h_G(x)_t)
where e denotes token embeddings. 

For a given position t, the discriminator predicts whether the token xt is âreal,â i.e., that it comes from the data rather than the generator distribution, with a sigmoid output layer.
$D(x, t) = sigmod(w^T h_D(x)_t)$

The generator then learns to predict the original identities of the masked-out tokens. 
The discriminator is trained to distinguish tokens in the data from tokens that have been replaced by generator samples. 
More specifically, we create a corrupted example x corrupt by replacing the masked-out tokens with generator samples and train the discriminator to predict which tokens in x corrupt match the original input x. 

loss function

L_{MLM}(x, \theta_G) = E(\sum_{i=m} -log p_G(x_i|x^{masked})) 
L_{Disc}(x, \theta_D) = E(\sum_{t=1}^n 1(x_t^{corrupt}=x_t) log D(x^{corrupt}, t) - 1(x_t^{corrupt}!=x_t) log (1 - D(x^{corrupt}, t)))

We minimize the combined loss
min_{\theta_G, \theta_D} \sum_{x \pert X} L_{MLM}(x, \theta_G) + \lambda L_{Disc}(x, \theta_D)

We approximate the expectations in the losses with a single sample. 
We donât back-propagate the discriminator loss through the generator (indeed, we canât because of the sampling step). 
After pre-training, we throw out the generator and fine-tune the discriminator on downstream tasks.

Our model architecture and most hyperparameters are the same as BERT. 
For fine-tuning on GLUE, we add simple linear classifiers on top of ELECTRA. 
For SQuAD, we add the questionanswering module from XLNet on top of ELECTRA, which is slightly more sophisticated than BERTâs in that it jointly rather than independently predicts the start and end positions and has a âanswerabilityâ classifier added for SQuAD 2.0. 
Some of our evaluation datasets are small, which means accuracies of fine-tuned models can vary substantially depending on the random seed. 
We therefore report the median of 10 fine-tuning runs from the same pre-trained checkpoint for each result. 

The proposed training objective jointly trains the generator and discriminator.

Smaller Generators:
If the generator and discriminator are the same size, training ELECTRA would take around twice as much compute per step as training only with masked language modeling. 
We suggest using a smaller generator to reduce this factor. 
Specifically, we make models smaller by decreasing the layer sizes while keeping the other hyperparameters constant.

================================================================================
Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
https://arxiv.org/abs/1901.02860
================================================================================

Despite the wide adoption, RNNs are difficult to optimize due to gradient vanishing and explosion (Hochreiter et al., 2001).
The introduction of gating in LSTMs and the gradient clipping technique (Graves, 2013) might not be sufficient to fully address this issue. 
Empirically, previous work has found that LSTM language models use 200 context words on average (Khandelwal et al., 2018), indicating room for further improvement.

Despite the success of transformers/attention, the LM training in Al-Rfou et al. (2018) is performed on separated fixed-length segments of a few hundred characters, without any information flow across segments. 
As a consequence of the fixed context length, the model cannot capture any longer-term dependency beyond the predefined context length. 
In addition, the fixed-length segments are created by selecting a consecutive chunk of symbols without respecting the sentence or any other semantic boundary. 
We refer to this problem as context fragmentation.

Our main technical contributions include:

* introducing the notion of recurrence in a purely selfattentive model and 
* deriving a novel positional encoding scheme. 

During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment.
this additional input allows the network to exploit information in the history.

* Check equations at "3.2 Segment-Level Recurrence with State Reuse".
    * SG() means stop gradients. Do not train the cached parameters.
    * The previous layer hidden state propagates to the next layer in the second equation, when calculating q, k, and v.
    * tao represents the cached data, tao + 1 the current embeddings.
    * First, the tao+1 embeddings for all layers are calculated, then they are cached.

* Figure 2: Illustration of the Transformer-XL model with a segment length 4.
    * Note that the cache can consist of M segments, not only 1.
    * Thus, we can cache a predefined length-M old hidden states spanning (possibly) multiple segments, and refer to them as the memory m, due to a clear connection to the memory augmented neural networks  (Graves et al., 2014; Weston et al., 2014).
    * The largest possible dependency length grows linearly w.r.t. the number of layers as well as the segment length, i.e., O(N x L), as visualized by the shaded area in Fig. 2b. In this example L=4.

* Relative Positional Encoding is explained in section 3.3 
    * Problem: how can we keep the positional information coherent when we reuse the states? 
    * If we apply the original transformer positional encoding strategy to the hidden states, both tao and tao+1 sequence embeddings are associated with the same positional encoding U1:L. 
      As a result, the model has no information to distinguish the positional difference between xĎ,j and xĎ+j. U is the positional encoding values.
    * By injecting the relative distance dynamically into the attention score, the query vector can easily distinguish the representations of x_tao,j and x_tao+1,j from their different distances, making the state reuse mechanism feasible. 
    * Matrix R defines relative positional encoding, where the i-th row R_i indicates a relative distance of i between two positions. 
    * 3 changes to the original transformer attention equations
        * replace all appearances of the absolute positional embedding Uj for computing key vectors in term (b) and (d) with its relative counterpart Riâj.
        * Introduce a trainable parameter u to replace the query U_i W_q in term (c), and similarly, a second term v.in term (d).
          Since the query vector is the same for all query positions, it suggests that the attentive bias towards different words should remain the same regardless of the query position.
        * separate the two weight matrices Wk,E and Wk,R for producing the content-based key vectors and location-based key vectors respectively.

================================================================================
CTRL: A Conditional Transformer Language Model for Controllable Generation
https://arxiv.org/abs/1909.05858
================================================================================

1.63 billion-parameter conditional transformer language model, trained to condition on control codes that govern style, content, and task-specific behavior. 

Because all control codes can be traced back to a particular subset of the training data, CTRL can be used to predict the subset of training data that is most likely given a sequence. 
This explicit relationship between CTRL and its training data can be exploited to analyze the correlations that the language model has learned from each domain.

These control codes also allow for the straightforward inclusion of task-specific data in a way that improves important skills without harming the generality of the model. 
Control codes for question answering and machine translation make these skills easily accessible.
These codes can be combined with codes during generation to create novel cross-over between control codes.

3. LANGUAGE MODELING WITH CTRL
    * The distribution can still be decomposed using the chain rule of probability and trained with a loss that takes the control code into account.
    * check equations
    * The control code c provides a point of control over the generation process. This is true even when sampling x_0.
    *training on sequences of raw text prepended with control codes. 
    * Each vector is the sum of a learned token embedding and a sinusoidal positional embedding as in the original Transformer architecture (Vaswani et al., 2017). 

* During training, these scores are the inputs of a cross-entropy loss function. 
* During generation, the scores corresponding to the final token are normalized with a softmax, yielding a distribution for sampling a new token.

* Training
    * We train on 140 GB 
    * We learn BPE (Sennrich et al., 2015) codes and tokenize the data using fastBPE
    * but we use a large vocabulary of roughly 250K tokens. 
    * preprocessing
        * we can filter out sequences that contain more than 2 unknown tokens. 
        * Data was treated as a single stream of tokens with non-domain control codes inserted where appropriate (often at document boundaries). 
        * The stream was chunked into contiguous sequences of tokens.  Each sequence originated from a domain
        * it has the corresponding domain control code prepended as the first token in the sequence to receive special treatment (Kobus et al., 2016). 
        * They are propagated to all text in the domain as the first token. 
        * All other control codes are injected into the data without such special treatment (Moryossef et al., 2019; Caswell et al., 2019). 
        * We experimented with sequence lengths of 256 and 512 due to memory and optimization constraints. 
    * CTRL has model dimension d = 1280, inner dimension f = 8192, 48 layers, and 16 heads per layer. 
    * Dropout with probability 0.1 follows the residual connections in each layer. 
    * Token embeddings were tied with the final output embedding layer.
    * global batch size of 1024 distributed across 256 cores of a Cloud TPU v3 Pod for 800k iterations. 
    * Training took approximately 2 weeks using Adagrad (Duchi et al., 2011) with a linear warmup from 0 to 0.05 over 25k steps. 
    * The norm of gradients were clipped to 0.25 as in (Merity et al., 2017). 
    * Learning rate decay was not necessary due to the monotonic nature of the Adagrad accumulator. 

* sampling (4.1)
    * Typically, temperature-controlled stochastic sampling methods are used for generating text from a trained language model. 
    * It is also common to limit the sampling only to the top-k alternatives.
    * p_i, probability of a word is softmax with temperature - see equation
        * When the temperature T approaches 0, it approximates a greedy distribution which magnifies the peaks in the probability distribution.
        * when T approaches infinity, it flattens the distribution to make it more uniform. 
    * Rather than choosing a fixed value of k, as is common practice, Holtzman et al. (2019) suggested adapting k heuristically. 
        * The nucleus sampling approach chooses a probability threshold p_t and sets k to be the lowest value such that sum(sort(p_i)) > p_t.
        * If the model is confident in its next-word prediction, then k will be lower and vice versa. 
    * we use near-greedy sampling but prevents repetitions through a penalty by discounting the scores of previously generated tokens. 
        * penalized sampling is not used during training. 
        * see equation - it is the temperature softmax adapted with a penalty term

================================================================================
Improving Language Understanding by Generative Pre-Training (GPT)
https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
================================================================================

gains on these tasks can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, followed by discriminative fine-tuning on each specific task. 

We employ a two-stage training procedure. 

1. we use a language modeling objective on the unlabeled data to learn the initial parameters of a neural network model. 
2. we adapt these parameters to a target task using the corresponding supervised objective.

During transfer, we utilize task-specific input adaptations derived from traversal-style approaches [52], which process structured text input as a single contiguous sequence of tokens. 

* We evaluate our approach on four types of language understanding tasks:
    * natural language inference,
    * question answering, 
    * semantic similarity, 
    * text classification. 

* Unsupervised pre-training (3.1)
    * uses transformer decoder
    * it seems a usual language model loss and transformer blocks - check formulas
    * trained using stochastic gradient descent [51].

* Supervised fine-tuning (3.2)
    * We assume a labeled dataset C, where each instance consists of a sequence of input tokens, x1, . . . , xm, along with a label y. 
    * The inputs are passed through our pre-trained model to obtain the final transformer blockâs activation h
    * which is then fed into an added linear output layer with parameters Wy to predict y:
    * We additionally found that including language modeling as an auxiliary objective to the fine-tuning helped learning
    * check formula for the final loss

* Task-specific input transformations (3.3)
    * We convert all structured inputs into token sequences to be processed by our pre-trained model, followed by a linear+softmax layer
    * For some tasks, like text classification, we can directly fine-tune our model as described above.
    * Certain other tasks, like question answering or textual entailment, have structured inputs such as ordered sentence pairs, or triplets of document, question, and answers. 
      Since our pre-trained model was trained on contiguous sequences of text, we require some modifications to apply it to these tasks.
    * we use a traversal-style approach [52], where we convert structured inputs into an ordered sequence that our pre-trained model can process. 
      These input transformations allow us to avoid making extensive changes to the architecture across tasks. 
    * Textual entailment task
        * we concatenate the premise p and hypothesis h token sequences, with a delimiter token ($) in between.
    * Similarity task
        * there is no inherent ordering of the two sentences being compared.
        * To reflect this, we modify the input sequence to contain both possible sentence orderings (with a delimiter in between) 
        * and process each independently to produce two sequence representations h, which are added element-wise before being fed into the linear output layer.
    * Question Answering and Commonsense Reasoning 
        * we are given a context document z, a question q, and a set of possible answers {ak}. 
        * We concatenate the document context and question with each possible answer, adding a delimiter token in between to get [z; q; $; ak]. 
        * Each of these sequences are processed independently with our model and then normalized via a softmax layer to produce an output distribution over possible answers.

* Model specifications 
    * Our model largely follows the original transformer work [62]. 
    * We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). 
    * For the position-wise feed-forward networks, we used 3072 dimensional inner states.
    * We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. 
    * The learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
    * We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
    * Since layernorm [2] is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient. 
    * We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53] and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. 
    * employed a modified version of L2 regularization proposed in [37], with w = 0.01 on all non bias or gain weights. 
    * For the activation function, we used the Gaussian Error Linear Unit (GELU) [18]. 
    * We used learned position embeddings instead of the sinusoidal version proposed in the original work.
    * We use the ftfy library to clean the raw text in BooksCorpus, standardize some punctuation and whitespace, and use the spaCy tokenizer

* Fine tuning
    * We add dropout to the classifier with a rate of 0.1. 
    * For most tasks, we use a learning rate of 6.25e-5 and a batchsize of 32. 
    * Our model finetunes quickly and 3 epochs of training was sufficient for most cases. 
    * We use a linear learning rate decay schedule with warmup over 0.2% of training. Îť was set to 0.5

* Natural Language Inference (NLI)
    * The task of natural language inference (NLI), also known as recognizing textual entailment, involves reading a pair of sentences and judging the relationship between them from one of entailment, contradiction or neutral. 

================================================================================
Language Models are Unsupervised Multitask Learners (GPT-2)
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
================================================================================

* is a 1.5B parameter Transformer
* ML systems need hundreds to thousands of examples to induce functions which generalize well. 
  This suggests that multitask training may need just as many effective training pairs to realize its promise with current approaches. 
* The current best performing systems on language tasks utilize a combination of pre-training and supervised finetuning. 
    * However, Recent work suggests that task-specific architectures are no longer necessary and transferring many self-attention blocks is sufficient (Radford et al., 2018) (Devlin et al., 2018).
* We demonstrate language models can perform down-stream tasks in a zero-shot setting â without any parameter or architecture modification. 

* Approach
    * Learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution p(output|input). 
    * Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. 
      That is, it should model p(output|input, task).  
      This has been variously formalized in multitask and meta-learning settings. 
    * a flexible way to specify tasks, inputs, and outputs all as a sequence of symbols. 
        * a translation training example can be written as the sequence (translate to french, english text, french text). 
        * a reading comprehension training example can be written as (answer the question, document, question, answer).
    * Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. 
        * If a language model is able to do this it will be, in effect, performing unsupervised multitask learning. 

* Input Representation (2)
    * A general language model (LM) should be able to compute the probability of (and also generate) any string. 
    * Current large scale LMs include pre-processing steps such as lowercasing, tokenization, and out-of-vocabulary tokens which restrict the space of model-able strings. 
    * Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences. 
    * We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . 
    * This results in a sub-optimal allocation of limited vocabulary slots and model capacity. 
    * To avoid this, we prevent BPE from merging across character categories for any byte sequence. 
    * We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.

* Model
    * follows the details of the OpenAI GPT model (Radford et al., 2018) 
    * few modifications.
        * Layer normalization (Ba et al., 2016) was moved to the input of each sub-block, similar to a pre-activation residual network (He et al., 2016) 
        * an additional layer normalization was added after the final selfattention block. 
        * A modified initialization which accounts for the accumulation on the residual path with model depth is used. 
        * We scale the weights of residual layers at initialization by a factor of 1/âN where N is the number of residual layers. 
        * The vocabulary is expanded to 50,257. 
        * We also increase the context size from 512 to 1024 tokens 
        * a larger batchsize of 512 is used.

================================================================================
BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension
https://arxiv.org/pdf/1910.13461.pdf
================================================================================

a denoising autoencoder for pretraining sequence-to-sequence models.
it pre-trains a model combining Bidirectional and Auto-Regressive Transformers. 

BART is trained by 
    1. corrupting text with an arbitrary noising function, and 
    2. learning a model to reconstruct the original text. 

The most successful approaches have been variants of masked language models, which are denoising autoencoders that are trained to reconstruct text where a random subset of the words has been masked out. 

A key advantage of this setup is the noising flexibility; arbitrary transformations can be applied to the original text, including changing its length. 
We evaluate a number of noising approaches, finding the best performance by both randomly shuffling the order of the original sentences and using a novel in-filling scheme, where arbitrary length spans of text (including zero length) are replaced with a single mask token. 

BART is particularly effective when fine tuned for text generation but also works well for comprehension tasks. 
It matches the performance of RoBERTa.

a) BERT: Random tokens are replaced with masks, and the document is encoded bidirectionally. Missing tokens are predicted independently, so BERT cannot easily be used for generation
b) GPT: Tokens are predicted auto-regressively, meaning GPT can be used for generation. However words can only condition on leftward context, so it cannot learn bidirectional interactions.
c) BART: Inputs to the encoder need not be aligned with decoder outputs, allowing arbitary noise transformations. Here, a document has been corrupted by replacing spans of text with mask symbols. The corrupted document (left) is encoded with a bidirectional model, and then the likelihood of the original document (right) is calculated with an autoregressive decoder.
   For fine-tuning, an uncorrupted document is input to both the encoder and decoder, and we use representations from the final hidden state of the decoder

* Model
    * It is implemented as a sequence-to-sequence model with a bidirectional encoder over corrupted text and a left-to-right autoregressive decoder. 
    * For pre-training, we optimize the negative log likelihood of the original document.
    * Follows the standard transformer seq-to-seq architecture except, following GPT, that we modify ReLU activation functions to GeLUs (Hendrycks & Gimpel, 2016)
    * initialise parameters from N (0, 0.02). 
    * For our base model, we use 6 layers in the encoder and decoder
    * For the large model, 12 layers in each.
    * Close to BERT architecture, with the following differences: 
        1. each layer of the decoder additionally performs cross-attention over the final hidden layer of the encoder (as in the transformer sequence-to-sequence model); and 
        2) BERT uses an additional feed-forward network before wordprediction, which BART does not. 
    * In total, BART contains roughly 10% more parameters than the equivalently sized BERT model.

* pre-training
    * BART is trained by corrupting documents and then optimizing a reconstruction lossâthe cross-entropy between the decoderâs output and the original document.
    * it allows us to apply any type of document corruption. 
    * In the extreme case, where all information about the source is lost, BART is equivalent to a language model.
    * transformations
        * Token Masking: like BERT
        * Token Deletion: Random tokens are deleted from the input. In contrast to token masking, the model must decide which positions are missing inputs
        * Text Infilling A number of text spans are sampled, with span lengths drawn from a Poisson distribution (Îť = 3). 
          Each span is replaced with a single [MASK] token. 
          0-length spans correspond to the insertion of [MASK] tokens. 
          Text infilling is inspired by SpanBERT (Joshi et al., 2019), 
          Text infilling teaches the model to predict how many tokens are missing from a span.
        * Sentence Permutation A document is divided into sentences based on full stops, and these sentences are shuffled in a random order.
        * Document Rotation A token is chosen uniformly at random, and the document is rotated so that it begins with that token.  This task trains the model to identify the start of the document.

* Fine-tuning BART (3)
    * Sequence Classification Tasks
        * the same input is fed into the encoder and decoder, 
        * the final hidden state of the final decoder token is fed into new multi-class linear classifier. 
        * This approach is related to the CLS token in BERT; however we add the additional token to the end so that representation for the token in the decoder can attend to decoder states from the complete input
    * Token Classification Tasks
        * such as answer endpoint classification for SQuAD, 
        * we feed the complete document into the encoder and decoder, 
        * use the top hidden state of the decoder as a representation for each word. 
        * This representation is used to classify the token
    * Sequence Generation Tasks
        * such as abstractive question answering and summarization.
        * information is copied from the input but manipulated, which is closely related to the denoising pre-training objective. 
        * Here, the encoder input is the input sequence, and the decoder generates outputs autoregressively.
    * Machine Translation
        * we replace BARTâs encoder embedding layer with a new randomly initialized encoder.
        * The model is trained end-to-end, which trains the new encoder to map foreign words into an input that BART can de-noise to English. 
        * The new encoder can use a separate vocabulary from the original BART model.
        * trained in 2 steps - in both cases backpropagating the cross-entropy loss from the output of the BART model. 
            1. we freeze most of BART parameters and only update the randomly initialized source encoder, the BART positional embeddings, and the self-attention input projection matrix of BARTâs encoder first layer. 
            2. we train all model parameters for a small number of iterations.

The effectiveness of pre-training methods is highly dependent on the task. 
For example, a simple language model achieves the best ELI5 performance, but the worst SQUAD results.

================================================================================
Multilingual Denoising Pre-training for Neural Machine Translation (mBART)
https://arxiv.org/abs/2001.08210
================================================================================

* mBART is the first method for pre-training a complete sequence-to-sequence model by denoising full texts in multiple languages,
* it pre-trains a complete autoregressive Seq2Seq model. mBART is trained once for all languages, providing a set of parameters that can be fine-tuned for any of the language pairs in both supervised and unsupervised settings, without any task-specific or language-specific modifications or initialization schemes.

* mBART enables new types of transfer across language pairs, for example, fine-tuning on bi-text in one language pair (e.g., Korean-English) creates a model that can translate from all other languages in the monolingual pre-training set (e.g., Italian-English), with no further training. 
* We also show that languages not in pre-training corpora can benefit from mBART, strongly suggesting that the initialization is at least partially language universal. 

* Data: CC25 corpus (2.1)
    * pre-train on a subset of 25 languages â CC25 â extracted from the Common Crawl (CC) (Wenzek et al., 2019; Conneau et al., 2019)
    * Following Lample and Conneau (2019), we rebalanced the corpus by up/down-sampling text from each language i with a ratio Îť_i - check formula
        * where p_i is the percentage of each language in CC25. 
        * We use the smoothing parameter Îą = 0.7.
    *Pre-processing 
        * We tokenize with a sentencepiece model (SPM, Kudo and Richardson, 2018)
        * this tokenization supports fine-tuning on additional languages. 
        * We do not apply additional preprocessing, such as truecasing or normalizing punctuation/characters.

* Model
    * Our models follow the BART (Lewis et al., 2019) sequence-to-sequence pre-training scheme
    * but we systematically study the effects of pre-training on different sets of languages
    * Architecture
        * 12 layers of encoder and 12 layers of decoder 
        * model dimension of 1024 on 16 heads 
        * 680M  parameters)
        * an additional layer-normalization layer on top of both the encoder and decoder, which we found stabilized training at FP16 precision.

* Learning
    * D = D1, ..., Dk. 
        * k languages
        * D_i is a collection of monolingual documents in language i
    * noising function g, that corrupts text, 
    * train the model to predict the original text X given g(X). Check formula
        * maximize log probability of X given g(X) for all X in each D_i

* Noise function Following Lewis et al. (2019), 
    * two types of noise in g. 
        * First remove spans of text and replace them with a mask token. 
          We mask 35% of the words in each instance by random sampling a span length according to a Poisson distribution (Îť = 3.5). 
        * permute the order of sentences within each instance. 
    * The decoder input is the original text with one position offset.
    * A language id symbol <LID> is used as the initial token to predict the sentence. 
    * It is also possible to use other noise types

* Instance format 
    * For each instance of a batch, we sample a language id symbol <LID>, and we pack as many consecutive sentences as possible sampled from the corresponding corpus of <LID>, until either it hits the document boundary or reaches the 512 max token length. 
    * Sentences in the instance are separated by the end of sentence (</S>) token. 
    * Then, we append the selected <LID> token to represent the end of this instance.
    * Pre-training at âmulti-sentenceâ level enables us to work on both sentence and document translation.

* Optimization 
    * 25 languages
    * trained on 256 Nvidia V100 GPUs (32GB) for 500K steps. 
    * The total batch size is around 128K tokens per GPU, matching BART (Lewis et al., 2019). 
    * We use the Adam optimizer ( = 1eâ6, Î˛2 = 0.98) 
    * linear learning rate decay scheduling. 
    * The total training time was approximately 2.5 weeks. 
    * We started the training with dropout 0.1 and reduced it to 0.05 at 250K steps and 0 at 400K steps. 
    * All experiments are done with Fairseq (Ott et al., 2019)

* Sentence-level Machine Translation (3)
    * Datasets We gather 24 pairs of publicly available parallel corpora that cover all the languages in CC25 (Table 1).
    * * We divide the datasets into three categories 
        * low resource (<1M sentence pairs), 
        * medium resource (>1M and <10M), 
        * high resource (>10M)

* Fine-tuning & Decoding 
    * We fine-tune our multilingual pre-trained models on a single pair of bitext data, feeding the source language into the encoder and decoding the target language. (figure 1)
    * we load the pre-trained weights and train the MT model on bi-texts with teacher forcing. 
    * For all directions, we train with 0.3 dropout, 0.2 label smoothing, 2500 warm-up steps, 3eâ5 maximum learning rate. 
    * We use a maximum of 40K training updates for all low and medium resource pairs and 100K for high resource pairs. 
    * For decoding, we use beam-search with beam size 5 for all directions. 
    * The final models are selected based on validation likelihood. 
    * The final results are reported in BLEU 

================================================================================
Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
https://arxiv.org/pdf/1910.10683.pdf
================================================================================

* introducing a unified framework that converts all text-based language problems into a text-to-text format. 
* By combining the insights from our exploration with scale and our new âColossal Clean Crawled Corpus, we achieve state-of-the-art results
* Crucially, the text-to-text framework allows us to directly apply the same model, objective, training procedure, and decoding process to every task we consider.
* We emphasize that our goal is not to propose new methods but instead to provide a comprehensive perspective on where the field stands.
 * We also explore the limits of current approaches by scaling up the insights from our systematic study (training models up to 11 billion parameters)
* to perform experiments at this scale, we introduce the âColossal Clean Crawled Corpusâ (C4), a data set consisting of hundreds of gigabytes of clean English text scraped from the web. 

* Setup (2)

* Overall, our encoder-decoder Transformer implementation closely follows its originallyproposed form (Vaswani et al., 2017). 
    * Layer normalization (Ba et al., 2016) is applied to the input of each subcomponent. 
    * We use a simplified version of layer normalization where the activations are only rescaled and no additive bias is applied. 
    * After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent?s input to its output.
    * Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack. 
    * The output of the final decoder block is fed into a dense layer with a softmax output, whose weights are shared with the input embedding matrix. 
    * While the original Transformer used a sinusoidal position signal or learned position embeddings, it has recently become more common to use relative position embeddings (Shaw et al., 2018; Huang et al., 2018a).
        * relative position embeddings produce a different learned embedding according to the offset between the key and query being compared in the self-attention mechanism. 
        * We use a simplified form of position embeddings where each embedding is simply a scalar that is added to the corresponding logit used for computing the attention weights. 
        * For efficiency, we also share the position embedding parameters across all layers in our model, though within a given layer each attention head uses a different learned position embedding. 
        * Typically, a fixed number of embeddings are learned, each corresponding to a range of possible key-query offsets. 
        * In this work, we use 32 embeddings for all of our models with ranges that increase in size logarithmically up to an offset of 128 beyond which we assign all relative positions to the same embedding. 
        * Note that a given layer is insensitive to relative position beyond 128 tokens, but subsequent layers can build a sensitivity to larger offsets by combining local information from previous layers.

* Model summary of differences to the original transformer
    * removing the Layer Norm bias
    * placing the layer normalization outside the residual path
    * using a different position embedding scheme.

* C4 (about 750 GB): Colossal Clean Crawled Corpus
    * we used the following heuristics for cleaning up Common Crawl?s web extracted text:
    * We only retained lines that ended in a terminal punctuation mark (i.e. a period, exclamation mark, question mark, or end quotation mark).
    * We discarded any page with fewer than 5 sentences and only retained lines that contained at least 3 words.
    * We removed any page that contained any word on the ?List of Dirty, Naughty, Obscene or Otherwise Bad Words?.6
    * we removed any line with the word Javascript.
    * Some pages had placeholder ?lorem ipsum? text; we removed any page where the phrase ?lorem ipsum? appeared.
    * Some pages inadvertently contained code. we removed any pages that contained a curly bracket.
    * To deduplicate the data set, we discarded all but one of any three-sentence span occurring more than once in the data set.
    * we used langdetect7 to filter out any pages that were not classified as English with a probability of at least 0.99. 



* tasks (2.3)
    * we measure performance on:
        * the GLUE and SuperGLUE text classification meta-benchmarks;
        * CNN/Daily Mail abstractive summarization; 
        * SQuAD question answering;
        * WMT English to German, French, and Romanian translation. 

* Input and Output Format (2.4)
    * we cast all of the tasks we consider into a text-to-text format
    * the model is trained with a maximum likelihood objective (using teacher forcing (Williams and Zipser, 1989)) regardless of the task. 
    * To specify which task the model should perform, we add a task-specific (text) prefix to the original input sequence before feeding it to the model.
    * check Figure 1 for example string formats
    * For text classification tasks (NLI), the model simply predicts a single word corresponding to the target label. 
        * Note that an issue arises if our model outputs text on a text classification task that does not correspond to any of the possible labels (for example if the model outputs?hamburger when the only possible labels for a task were entailment, neutral, or contradiction). 
        * In this case, we always count the model?s output as wrong, though we never observed this behavior in any of our trained models. 
    * Note that the choice of text prefix used for a given task is essentially a hyperparameter; we found that changing the exact wording of the prefix had limited impact and so did not perform extensive experiments into different prefix choices. 
    * We were able to straightforwardly cast all of the tasks we considered into a text-to-text format with the exception of STS-B, which is a regression task where the goal is to predict a similarity score between 1 and 5. 
        * * We found that most of these scores were annotated in increments of 0.2, so we simply rounded any score to the nearest increment of 0.2 
        * converted the result to a literal string representation of the number 
        * At test time, if the model outputs a string corresponding to a number between 1 and 5, we convert it to a floating-point value; otherwise, we treat the model?s prediction as incorrect. 
        * This effectively recasts the STS-B regression problem as a 21-class classification problem.

* inspired in previous work 
    * McCann et al. (2018) propose the Natural Language Decathlon
        * a benchmark that uses a consistent question-answering format for a suite of ten NLP tasks.
        * it stipulates that all models must be multi-task, i.e. are able to simultaneously tackle all of the tasks at once. 
        * We instead allow for separately fine-tuning the model on each individual task and 
        * use short task prefixes instead of an explicit question-answer format.
    *  Radford et al. (2019) evaluate the zero-shot learning capabilities of language models by feeding some input to the model as a prefix and then autoregressively sampling an output. 
        * For example, automatic summarization is done by feeding in a document followed by the text ?TL;DR:? (short for ?too long, didn?t read?, a common abbreviation) and then the summary is predicted via autoregressive decoding. 
        * We mainly consider models that explicitly process an input with an encoder before generating an output with a separate decoder and 
        * we focus on transfer learning rather than zero-shot learning. 
    * Keskar et al. (2019b) unify many NLP tasks as span extraction
        * where text corresponding to possible output choices are appended to the input and the model is trained to extract the input span corresponding to the correct choice. 
        * In contrast, our framework also allows for generative tasks like machine translation and abstractive summarization where it is not possible to enumerate all possible output choices.

* training
    * always train using standard maximum likelihood, i.e. using teacher forcing (Williams and Zipser, 1989) and a cross-entropy loss
    * For optimization, we use AdaFactor (Shazeer and Stern, 2018). 
    * At test time, we use greedy decoding (i.e. choosing the highest-probability logit at every timestep).
    * We pre-train each model for 2^19 = 524,288 steps on C4 before fine-tuning. 
    * We use a maximum sequence length of 512 and a batch size of 128 sequences. 
    * Whenever possible, we?pack multiple sequences into each entry of the batch10 so that our batches contain roughly 2^16 = 65,536 tokens. 
    * In total, this batch size and number of steps corresponds to pre-training on 2^35 ? 34B tokens. 
    * This is considerably less than BERT (Devlin et al., 2018), which used roughly 137B tokens, or RoBERTa (Liu et al., 2019c), which used roughly 2.2T tokens. 
    * During pre-training, we use an ?inverse square root? learning rate schedule:
        * This sets a constant learning rate of 0.01 for the first 104 steps, then exponentially decays the learning rate until pre-training is over. 
    * We also experimented with using a triangular learning rate (Howard and Ruder, 2018), which produced slightly better results but requires knowing the total number of training steps ahead of time. 
    * During fine-tuning
        * fine-tuned for 2^18 = 262,144 steps on all tasks. 
        * we continue using batches with 128 length-512 sequences  (i.e. 2^16 tokens per batch). 
        * We use a constant learning rate of 0.001 
        * We save a checkpoint every 5,000 steps
        * report results on the model checkpoint corresponding to the highest validation performance. 

* vocabulary
    * We use SentencePiece (Kudo and Richardson, 2018) to encode text as WordPiece tokens (Sennrich et al., 2015; Kudo, 2018). 
    * For all experiments, we use a vocabulary of 32,000 wordpieces. 
    * This vocabulary was shared across both the input and output of our model. 
    * Note that our vocabulary makes it so that our model can only process a predetermined, fixed set of languages.

* objective
     * it has recently been shown that ?denoising? objectives (Devlin et al., 2018; Taylor, 1953) (also called ?masked language modeling?) produce better performance than causal language modeling objective for pre-training.
    * Corrupting Spans (3.3.4) is slightly better than masking with better performance by predicting shorter targets.
     * The approach we have used so far makes an i.i.d. decision for each input token as to whether to corrupt it or not. 
     * When multiple consecutive tokens have been corrupted, they are treated as a ?span? and a single unique mask token is used to replace the entire span.
     * Corrupting spans was also previously considered as a pre-training objective for BERT, where it was found to improve performance (Joshi et al., 2019).

================================================================================
ERNIE: Enhanced Language Representation with Informative Entities
https://arxiv.org/abs/1905.07129
================================================================================



================================================================================
PMI-MASKING: PRINCIPLED MASKING OF CORRELATED SPANS
https://arxiv.org/abs/2010.01825
================================================================================

 * In BERT, 15% of tokens are chosen to be masked uniformly at random. 
 * It is the random choice of single tokens that we address in this paper: we show that this approach is suboptimal and offer a principled alternative.

* The advantage of Whole-Word Masking over Random-Token Masking is relatively modest for standard vocabularies, because out-of-vocabulary words are rare. 
* However, the tokenization of words is a very special case of a much broader statistical linguistic phenomenon of collocation: 
    * the cooccurrence of series of tokens at levels much greater than would be predicted simply by their individual frequencies in the corpus. 
    * There are millions of collocated word n-grams ? multi-word expressions, phrases, and other common word combinations?whereas there are only tens of thousands of words in frequent use. 

* Several prior works have considered the idea of masking across spans longer than a single word.
    * Sun et al. (2019) proposed Knowledge Masking 
        * which jointly masks tokens comprising entities and phrases, as identified by external parsers. 
        * While extending the scope of Whole-Word Masking, the restriction to specific types of correlated n-grams, along with the reliance on imperfect tools for their identification, has limited the gains achievable by this approach. 
    * SpanBERT of Joshi et al. (2020) introduced Random-Span Masking
        * which masks word spans of lengths sampled from a geometric distribution at random positions in the text. 
        * it was shown to consistently outperform Knowledge Masking, is simple to implement, and inspired prominent MLMs (Raffel et al., 2019). 
        * However, while Random-Span Masking increases the chances of masking collocations, with high probability the selected spans include only part of a collocation along with unrelated neighboring tokens, potentially wasting resources on spans that provide little signal.

* EXISTING MASKING APPROACHES (3.1)
    * Random-Token Masking (Devlin et al., 2019a) The original BERT implementation 
        * selects tokens for masking independently at random
        * where 80% of the 15% chosen tokens are replaced with [MASK], 10% are replaced with a random token, and 10% are kept unchanged.
    * Whole-Word Masking (Devlin et al., 2019b) 
        * The sequence of input tokens is segmented into units corresponding to whole words. 
        * Tokens for masking are then chosen by sampling entire units at random until the masking budget is met. 
        * Following Devlin et al. (2019a), for 80%/10%/10% of the units, all tokens are replaced with [MASK]tokens/ random tokens/ the original tokens, respectively.
    * Random-Span Masking (Joshi et al., 2020) 
        * Contiguous random spans are selected iteratively until the 15% masking budget is spent. 
        * At each iteration, a span length (in words) is sampled from a geometric distribution Geo(0:2), and capped at 10 words. 
        * Then, the starting point for the span to be masked is randomly selected. 
        * Replacement with [MASK], random, or original tokens is done as above, where spans constitute the units.

* PMI: FROM BIGRAMS TO n-GRAMS (3.2)
    * modeling such correlations in large corpora was widely studied in computational linguistics (Zuidema (2006); Ramisch et al. (2012); inter alia). 
    * Particularly relevant to our work is the notion of Pointwise Mutual Information (Fano, 1961), which quantifies how often two events occur, compared with what we would expect if they were independent. 
    * Define the probability of any n-gram as the number of its occurrences in the corpus divided by the number of all the n-grams in the corpus. 
    * PMI leverages these probabilities to give a natural measure of collocation of bigrams: how surprising the bigram w1w2 is, given the unigram probabilities of w1 and w2. 
    * formula (mutual information) - compares the actual empirical probability of the n-gram in the corpus with the probability it would have if its components occurred independently. (ratio)
    * problem: an n-gram's Naive-PMI will be high if it contains a segment with high PMI, even if that segment is not particularly correlated with the rest of the n-gram. 
    * solution: Intuitively, this measure effectively discards the contribution of high PMI segments; the minimum in Eq. 3 implies that an n-gram?s collocation score is given by its weakest link, i.e., by the segmentation that is closest to separability. 

* PMI MASKING - MASKING CORRELATED n-GRAMS
    * assembling a list of n-grams as a masking vocabulary in parallel to the 30K-token vocabulary. 
    * we make use of the entire pretraining corpus for compiling a list of collocations. 
    * We consider word n-grams of lengths 2?5 having over 10 occurrences in the corpus, and include the highest ranking collocations over the corpus, as measured via our proposed PMIn measure (Eq. 3). 
    * Noticing that the PMIn measure is sensitive to the length of the n-gram, we assemble per-length rankings for each n in {2; 3; 4; 5}, and integrate these rankings to compose the masking vocabulary. 
    * we chose the masking vocabulary size to be 800K, for which approximately half of pretraining corpus tokens were identified as part of some correlated n-gram.
    * After composing the masking vocabulary, we treat its entries as units to be masked together. 
    * All input tokens not identified with entries from the masking vocabulary are treated independently as units for masking according to the Whole-Word Masking scheme. 
    * After we segment the sequence of input tokens into units for masking, we then choose tokens for masking by sampling units uniformly at random until 15% of the tokens (the standard tokens of the 30K-token vocabulary) in the input are selected. 
    * As in the prior methods, replacement with [MASK](80%), random (10%), or original (10%) tokens is done at the unit level.

* *PRETRAINING
    * We trained uncased models with a 30K-sized vocabulary that we constructed over WIKIPEDIA +BOOKCORPUS via the WordPiece Tokenizer used in BERT. 
    * We omitted the Next Sentence Prediction task, as it was shown to be superfluous (Joshi et al., 2020),
    * trained only on the Masked Language Model task during pretraining. 
    * We trained with a sequence length of 512 tokens, batch size of 256, and a varying number of steps detailed in Section 5. 
    * For pretraining, after a warmup of 10; 000 steps we used a linear learning rate decay, therefore models that ran for a different overall amount of steps are not precisely comparable after a given amount of steps. 
    * We set remaining parameters to values similar to those used in the original BERT pretraining, detailed in the appendix.

* We show that PMI-Masking achieved even larger performance gains relative to the baselines when training over more data, by adding the 38GB OPENWEBTEXT (Gokaslan & Cohen, 2019) dataset, an open-source recreation of the WebText corpus described in Radford et al. (2019).

================================================================================
On Losses for Modern Language Models
https://arxiv.org/abs/2010.01694
================================================================================

* We show that NSP is detrimental to training due to its context splitting and shallow semantic signal. 
* We also identify six auxiliary pre-training tasks: sentence ordering, adjacent sentence prediction, TF prediction, TF-IDF prediction, a Fast- Sent variant, and a Quick Thoughts variant ? that outperform a pure MLM baseline. 
* Finally, we demonstrate that using multiple tasks in a multi-task pre-training framework provides better results than using any single auxiliary task. 
* Using these methods, we outperform BERTBase on the GLUE benchmark using fewer than a quarter of the training tokens.

* Baseline
    * For computational reasons we use BERTBase (L = 12, H = 768, A = 12, Total Parameters=110M), 
    * use the uncased WordPiece tokenizer (Wu et al., 2016) with vocabulary size of 30522 provided by Google?.

* Token level tasks
    1. Term Frequency prediction (TF): Regression predicting a token?s frequency in the rest of the document. The frequency is re-scaled between 0 and 10 per document.
    2. Term Frequency-Inverse Document Frequency prediction (TF-IDF): Regression predicting a token?s tf-idf that has been re-scaled between 0 and 10 per document.
    3. Sentence Boundary Objective (SBO): Predict the masked token given the embeddings of the adjacent tokens.
    4. Trigram-Shuffling (TGS): 6-way classification predicting the original order of shuffled tri-grams
    5. Token Corruption Prediction (TCP): Binary classification of whether a token has been corrupted (inserted, replaced, permuted) or not.
    6. Capitalization Prediction (Cap.): Binary, whether a token is capitalized or not.
    7. Token Length Prediction (TLP): Regression to predict the length of the WordPiece token.
* Sentence level tasks
    8. Next Sentence Prediction (NSP): Binary, whether the second sentence follows the first or comes from a separate document.
    9. Adjacent Sentence Prediction (ASP): 3-way classification whether the second sentence proceeds the first, precedes the first, or they come from separate documents.
    10. Sentence Ordering (SO): Binary, predicting if the two sentences are in or out of order.
    11. Sentence Distance Prediction (SDP): 3-way classification of whether the second sentence proceeds, the two sentences are noncontiguous from the same document, or come from separate documents.
    12. Sentence Corruption Prediction (SCP): Binary classification of whether a tokens in a sentence have been corrupted (inserted, replaced, permuted) or not.
    13. Quick Thoughts variant (QT): Split each batch into two, where the second half contains the subsequent sentences of the first half (e.g. with batch size 32, sentence 17 follows sentence 1, sentence 18 follows sentence 2,...). We use an energy-based model to predict the correct continuation for each sentence in the first half where the energy between two sentences is defined by the negative cosine similarity of their [CLS] embeddings. (figure 1)
    14. FastSent variant (FS): same as QT above. The loss is defined as cross-entropy between 1.0 and the cosine similarity of a sentence [CLS] embedding and the other sentence token embeddings ([CLS] embedding from the first half with token embeddings from the second half and [CLS] embeddings from second half with token embeddigns from the first half). We use one model to encode both halves concurrently.

* Combining tasks (3.3)
    * BERT originally proposed summing the MLM and NSP losses directly. 
    * ERNIE uses significantly more losses and proposes a continual multi-task learning framework to incorporate them, in which they incrementally add new tasks while sampling previously learnt tasks. 
    * we investigate the six following ways of combining a set of tasks for BERT pre-training:
        1. Sum losses from all tasks (sum.)
        2. Incrementally add tasks, summing the losses         from all added tasks (Inc.)
        3. Alternating between tasks each iteration (Alt.) 
        4. Alternating between auxiliary tasks each iteration and summing it with MLM (Alt.+)
        5. ERNIE?s continual multi-task learning (CMTL), for more detail see Appendix A
        6. ERNIE?s continual multi-task learning on auxiliary tasks summed with MLM (CMTL+)

* Input Representation (3.4)
    * we sum token embeddings, learned position embeddings, learned sentence type (sentence A or B) embeddings, and, to enable ERNIE?s continual multi-task learning, a learned task id embeddings .

* Dataset (3.5)
    * We follow precedent in using the BookCorpus? (Zhu et al., 2015) and Wikipedia dataset as our corpora.
    * We filter the Wikipedia corpus in the same fashion as BERT, ignoring lists, tables, and headers.
    * We additionally filter documents that have: fewer than 10 words or fewer than 4 sentences. 
    * We additionally segment long documents into documents of roughly 1024 tokens. 
    * This creates a corpus with 2.7 billion words (3.8 billion tokens) divided into 6.8 million documents.

* Pre-Training Details (3.6)
    * For all tests, we train on 10 billion tokens
    * Adam optimizer (Kingma and Ba, 2014) 
    * learning rate of 1e-4 that warms-up over the first 1% of tokens and linearly decays after
    * batch size = 128, max sequence length = 128, 
    * beta_1 = 0.9, beta_2 = 0.999, L2 weight decay of 0.01, 
    * dropout probability of 0.1. 
    * gelu activation (Hendrycks and Gimpel, 2016). 
    * Using four p100 GPUs, it takes between 13 and 15 hours to train a model for each one billion token epoch depending on the tasks used.

* Fine-Tuning Details (3.7)
    * All models are tested on the GLUE (Wang et al., 2018) benchmark, and SuperGLUE (Wang et al., 2019a) benchmark. 
    * Following Devlin et al. (2018); Cheng et al. (2019), we disregard GLUE’s problematic WNLI task. 
    * To fine-tune the model on the GLUE dataset, we use Jiant’s (Wang et al., 2019b) provided code. 
    * We limit the maximum number of epochs to 3 
    * we run the fine-tuning procedure three times with learning rates = 5e-5, 3e-5, 2e-5 and take the best results for each task individually
    * For all other fine-tuning parameters, we use the default values provided by jiant unless otherwise stated.

* Understanding NSP (4.1)
    * splitting the context imposes inherent limitations on language models.

* Auxiliary Tasks (4.2)
    * We first compare the 14 auxiliary tasks in Table 1 to a MLM baseline (No Aux.). 
    * NSP is detrimental to training. It provides a shallow supervision signal, and is often solvable through lexical overlap. 
    * Adjacent sentence prediction and sentence ordering on the other hand require deeper semantic understanding of the structure of language. 
        * with SO and ASP outperforming MLM and NSP on all inference tasks and greatly outperforming all auxiliary tasks on RTE, the only low-resource inference task. 
    * The model trained using the Quick Thoughts variant (QT) performs the best out of all the above models. 
        * We hypothesize that the loss, based on cosine similarity, provides a soft clustering around semantically similar topics, which produces more distinguishable embeddings. 
    * The FastSent variant (FS) provides a similar signal and performs the second best, suggesting that some form of soft clustering does provide substantial benefit to pre-training. 
    * TF-IDF, and to a lesser extent TF, prediction also improve performance on a range of downstream tasks. 
        * This aligns with Sun et al. (2019b)’s observations that identifying high value words (and discounting filler words) provides a useful signal for language models. 
    * All other tasks fail to provide any meaningful gains. 
    * Our results did not find the Sentence Boundary Objective (SBO) to be beneficial. 
        * However, as it was originally implemented for spans, this does not discount the results of Joshi et al. (2019); in our context, which only masks a single word, it is likely redundant with MLM. 

* Combining Tasks (4.3)
    * To test combining multiple tasks, we use all auxiliary losses that substantially outperform a pure MLM baseline. 
    * For tasks that provide similar signals, we select the one that achieved a higher average
    * This provides 4 tasks for the multi-task training: MLM, QT, SO, and TF-IDF.
    * Our results indicate that multitask training with MLM preserves the benefits of each individual task, with the combined models retaining QT's high CoLA score and SO's high RTE score. 
    * Further, these gains are additive in most cases: for QNLI, MNLI, and STS-B the combined models performs better than any single auxiliary task models. 
    * Between combination methods that use MLM in every iteration, the incremental approach appears to be the worse, 
    * while summing everything, alternating auxiliary tasks (Alt.+), and continual multi-task learning on auxiliary tasks (CMTL+) all perform similarly, 
    * Interestingly, both approaches where tasks vary each iteration (Alt.+ and CMTL+) see a significant benefit on the CoLA task. 
        * While not beneficial in our framework, an alternating pattern or CMTL have the additional benefit of enabling different input structures or the use of different corpora 

* Final Results (4.4)
    * For our final test, we train our baselineMLMmodel and CMTL+ model on 32 billion tokens and present the results using the GLUE and SuperGLUE evaluation
    * When fine-tuning these models, we run an exhaustive hyper parameter search on 
        * learning rates = 1e-5, 2e-5, 3e-5, 5e-5, 
        * batch sizes = 16, 32, 
        * number of epochs = 2, 3, 4. 
    * The results show that the CMTL+ model - trained on MLM, QT, SO, and TF-IDF in a continual multi-task learning framework - vastly outperforms the MLM baseline in every task. 
    * Further, our model trained on 32 billion tokens outperforms the original BERTBase, which required 137 billion tokens.

* Discussion (5)
    * NSP prediction is a semantically shallow and often solvable through lexical overlap 
    * using a task that requires understanding the ordering of contiguous text provides a stronger semantic signal;
    * a language model should be trained in a multi-task setting. 
    * Providing a signal to reduce the embedding distance between semantically similar sentences, as in our FastSent or QuickThought variants 
    * Providing a signal that relays word importance, such as TF-IDF and TF, likewise produces substantial benefit to BERT pre-training.
    * We show strong evidence that a MLM variant loss should always be included when multi-task learning.
    * combining multiple beneficial tasks leads to better results than using any of the individual tasks alone.


