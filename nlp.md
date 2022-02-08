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
* We demonstrate language models can perform down-stream tasks in a zero-shot setting without any parameter or architecture modification. 

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

* Dataset
    * we created a new web scrape which emphasizes document quality. 
    * we only scraped web pages which have been curated/filtered by humans. 
    * we scraped all outbound links from Reddit, a social media platform, which received at least 3 karma. 
    * The resulting dataset, WebText
    * contains the text subset of these 45 million links. 
    * All results presented in this paper use a preliminary version of WebText which does not include links created after Dec 2017 
    * which after de-duplication and some heuristic based cleaning contains slightly over 8 million documents for a total of 40 GB of text. 
    * We removed all Wikipedia documents from WebText since it is a common data source for other datasets and could complicate analysis due to over lapping training data with test evaluation tasks.

* Input Representation (2)
    * A general language model (LM) should be able to compute the probability of (and also generate) any string. 
    * Current large scale LMs include pre-processing steps such as lowercasing, tokenization, and out-of-vocabulary tokens which restrict the space of model-able strings. 
    * Byte Pair Encoding (BPE) (Sennrich et al., 2015) is a practical middle ground between character and word level language modeling which effectively interpolates between word level inputs for frequent symbol sequences and character level inputs for infrequent symbol sequences. 
    * We observed BPE including many versions of common words like dog since they occur in many variations such as dog. dog! dog? . 
    * This results in a sub-optimal allocation of limited vocabulary slots and model capacity. 
    * To avoid this, we prevent BPE from merging across character categories for any byte sequence. 
    * We add an exception for spaces which significantly improves the compression efficiency while adding only minimal fragmentation of words across multiple vocab tokens.
    * Since our approach can assign a probability to any Unicode string, this allows us to evaluate our LMs on any dataset regardless of pre-processing, tokenization, or vocab size.

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

================================================================================
Muppet: Massive Multi-task Representations with Pre-Finetuning
https://arxiv.org/abs/2101.11038
================================================================================

* We propose pre-finetuning, an additional largescale learning stage between language model pre-training and fine-tuning. 
* Pre-finetuning is massively multi-task learning (around 50 datasets, over 4.8 million total labeled examples),
* it consistently improves performance for pretrained discriminators (e.g. RoBERTa) and generation models (e.g. BART) 
* pre-finetuning can hurt performance when few tasks are used up until a critical point (usually above 15) after which performance improves linearly in the number of tasks.
* We show that standard multi-tasking schemes can be unstable and often fail to learn high quality representations. 
    * However, we introduce a new training scheme which uses loss scaling and task-heterogeneous batches so that gradient steps are more evenly balanced across multiple different competing tasks, 

* Pre-Finetuning Through Massive Multitask Learning (3)
    * it can be challenging to balance the losses from different tasks; 
    * upsampling can lead to overfitting low resource tasks,
    * and downsampling can lead to improper learning of specific tasks. 
    * This difficulty is particularly pronounced when operating at the scale 

* Tasks and Losses (3.1)
    * We select language tasks across four different domains: classification, commonsense reasoning, machine reading comprehension, and summarization.
    * In total our multi-task set up learns over 4.8M supervised samples across 4 families of tasks.
    * Standard Losses
        * our model contains task-specific heads, each optimizing for a task-specific loss. 
            * Classification: Cross Entropy (CE)
            * Summarization: Label Smoothed CE (Szegedy et al., 2015)
            * MRC: Span Prediction (Seo et al., 2016)
            * Commonsense: Sentence Ranking Loss (Liu et al., 2019b)
        * Each loss is scaled with loss scaling described in �3.3. 
        * After loss scaling, the gradients from each task are averaged before doing the model update step.

* Optimization (3.2)
    * We show two strategies to learn multi-task representations at scale
    * Accumulating Gradients Across Tasks (Heterogeneous Batches) 
        * model is trying to optimize not a single objective but several potentially competing objectives
        * moving along the gradient of a single task may not be the optimal direction for the model to move to learn a single unified representation 
        * To overcome this, we ensure each batch our model optimizes consists of several tasks. 
        * Each worker samples a random batch from our set of tasks and computes a gradient, accumulated for the final update. 
        * Empirically we use 64 GPUs for pre-finetuning, resulting in each batch consisting of gradients across 64 sampled tasks. 
    * Leveraging Better Finetuning.
        * learned from self-supervised pre-training in pre-finetuning. 
        * Mosbach et al. (2020) show that standard fine-tuning of pre-trained models can be unstable, which may be aggravated in our case as we are training on a diverse set of tasks simultaneously.
        * Therefore, we employ the R3F/R4F methods (Aghajanyan et al., 2020) 
        * consists of an additional loss term, ensuring that small perturbations to the input space result in similar representations, which can be used to learn more robust representations during pre-finetuning.

* Loss Scaling (3.3)
    * introduce a multiplicative reweighting of individual losses per data-point. 
    * As pre-finetuning optimizes several different types of tasks and datasets, each having its own output spaces, loss scaling becomes essential to ensure stable training. 
    * We scale data-point loss so that, if the class distribution were uniformly distributed along with our models predictions, all of our losses would have equivalent values. - check equation (1)

* Sampling (3.4)
    * Another approach to balancing various tasks in a multi-task set up is to up-sample smaller datasets and down-sample larger ones to achieve more uniformity between dataset sizes.
    * recent work has shown that it does not work well for multitask learning of pre-trained representations (for example, T5)
    * We also found that sampling datasets were consistently detrimental for multi-task learning over pre-trained representations during initial experimentation.

* Experimental Setup (3.5)
    * *RoBERTa (Liu et al., 2019b) and BART (Lewis et al., 2019) as our initial pre-trained models to further pre-finetune. 
    * For each task type we use a different prediction scheme. 
    * Every Sentence Prediction dataset gets a separate classification head
    * for Commonsense and MRC we utilize a separate unified head for each task. 
    * For Summarization, we do not add any parameters and use the BART decoder and output layer as is. 
    * Experimentally we saw using a different head per individual Commonsense and MRC datasets lead to severe overfitting.
    * We trained each model configuration with 64 GPUs until convergence (this ranged from a day to 4 days)

* Experiments (4)
    * Given that our pre-finetuned models now have an understanding of the task at hand through the use of classification heads, we have a choice during finetuning on whether or not to use these heads. 
    * In general we found re-using heads to be beneficial for MRC, Commonsense and Sentence Prediction tasks with small dataset size.
    * We see more modest gains on larger datasets, most likely because we do not need to refine representations beforehand if the fine-tuning dataset is large.
    * On smaller datasets, we see substantial gains. 
    * For example, the pre-finetuned RoBERTa-BASE model on RTE improves by close to 9 points, rivaling the RoBERTa-Large accuracy, 
    * while the pre-finetuned RoBERTa-Large model gets new state-of-the-art on RTE rivaling models an order of magnitude larger than it.
    * Our methods do not increase parameter count or any complexity measures but are quite successful at refining features and preparing them for downstream fine-tuning.

* Understanding Multi-Task at Scale (5)

* Importance of Scale
    * T5 and MT-DNN, focused on the MTL scale of around a dozen datasets. To the best of our knowledge, our paper has the largest MTL set up to date. 
    * train seven models, 
    * six uniformly chosen between 10 and 40, ensuring that at each point, the selected datasets are a superset of the datasets from prior points. 
    * The last model is fully trained on all datasets. 
    * For each version of the model, we finetune STS-B (Cer et al., 2017), BoolQ (Clark et al., 2019), RACE (Lai et al., 2017), SQuAD (Lai et al., 2017), and MNLI (Williams et al., 2018a).
    *  We include these five datasets in the first MTL run (10 datasets) to remove any bias from adding them in a later stage.
    * We see a couple of interesting patterns. 
        * for individual tasks such as RTE (Bentivogli et al., 2009), increasing the pre-finetuning scale monotonically improves performance. 
            * This is aligned with other papers that have seen benefits from first training on MNLI (Williams et al., 2018a) and then fine-tuning on RTE (Liu et al., 2019b). 
        * For other datasets, we see that doing MTL in the < 15 datasets regime is detrimental for end-task finetuning.
            * This is also aligned with other empirical observations, i.e., T5 reported that doing MTL did not improve over only fine-tuning.  
        * Nevertheless, it seems that as we increase the number of tasks past some critical point, our pre-trained representations become more generalizable. 
        * although dependent on the dataset, this critical point is roughly between 10 and 25 tasks.
        * This suggests that previously observed MTL limitations were not fundamental and can instead be attributed to the lack of sufficient scale.


* Importance of Heterogenous Batches (5.2)
    * critical aspect is the method through which MTL is implemented, specifically the selection of batches. 
    * we experimented with three balancing schemes:
        * dataset homogenous
            * selecting batches from datasets sequentially. 
            * first train on dataset A, then train on dataset B, etc.
        * batch homogenous 
            * selecting batches containing only data from the  same task; therefore, all gradients are from thesame dataset. 
            * This is implemented by selecting all  datasets, batching on a dataset level, and selectingthose same batches randomly during training.
        * batch heterogenous
            * a single update containing a batch from multiple different datasets spanning different tasks. 
            * We implemented this by first creating homogenous sub-batches, calculating loss per sub-batch per GPU, and then aggregating across GPUs manifesting in a gradient update that contains various datasets and, therefore, tasks.
    * Our findings are also consistent with (Aghajanyan et al., 2020) which saw that sequential training of data-sets degrades generalizable representations.

* Low Resource Experiments (5.3)
    * We noticed in Section �4 that data-sets with smaller data-set sizes tended to improve more from MTL
    * we look at two factors: the scale of pre-finetuning and the scale of fine-tuning (size of fine-tuning data-set).
    * We select three data-sets that were not used in pre-finetuning in Section �5.1. 
    * We also select nine partitions per fine-tuning data-set, which is sampled uniformly between 10% of the data-set and 100% of the data-set. 
    * Selecting the low-resource splits was done through random sampling.
    * We then fine-tune every low-resource split with every pre-finetuning checkpoint from Section �5.1.
    * As we increase the scale of MTL, better representations are available for downstream finetuning. 
    * Furthermore, we see that prefinetuned models at a larger scale are much more data-efficient than standard pre-trained models.

================================================================================
Language Models are Few-Shot Learners (GPT3)
https://arxiv.org/abs/2005.14165
================================================================================

* NLP usually pre-training + fine-tuning on a specific task. 
* While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. 
* By contrast, humans can generally perform a new language task from only a few examples or from simple instructions 
* GPT-3, an autoregressive language model with 175 billion parameters, 
* test its performance in the few-shot setting. 
* For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model.

* introduction (1)
    * meta-learning: 
        * which in the context of language models means the model develops a broad set of skills and pattern recognition abilities at training time, and then uses those abilities at inference time to rapidly adapt to or recognize the desired task (illustrated in Figure 1.1). 
        * In the context of language models this has sometimes been called ?zero-shot transfer?, but this term is potentially ambiguous: the method is ?zero-shot? in the sense that no gradient updates are performed, but it often involves providing inference-time demonstrations to the model, so is not truly learning from zero examples. 
        * we use the term ?meta-learning? to capture the inner-loop / outer-loop structure of the general method, 
        * and the term ?in context-learning? to refer to the inner loop of meta-learning. 
        * We further specialize the description to ?zero-shot?, ?one-shot?, or ?few-shot? depending on how many demonstrations are provided at inference time. 
    * RWC+ [19] attempts to do this via what we call ?in-context learning?, 
        * uses the text input of a pretrained language model as a form of task specification: 
        * the model is conditioned on a natural language instruction and/or a few demonstrations of the task and is then expected to complete further instances of the task simply by predicting what comes next.
        * While it has shown some initial promise, this approach still achieves results far inferior to fine-tuning

* approach (2)
    * Our basic pre-training approach, including model, data, and training, is similar to the process described in [RWC+19],
    * our use of in-context learning is also similar to [RWC+19]
    * check figure 2.1 **important** input format for the model

* Model and Architectures (2.1)
    * We use the same model and architecture as GPT-2 [RWC+19], 
    * including the modified initialization, pre-normalization, and reversible tokenization described therein, 
    * with the exception that we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer [CGRS19]. 
    * we train 8 different sizes of model, ranging over three orders of magnitude from 125 million parameters to 175 billion parameters, with the last being the model we call GPT-3. 
    * we always have the feedforward layer four times the size of the bottleneck layer
    * All models use a context window of nctx = 2048 tokens. 
    * We partition the model across GPUs along both the depth and width dimension in order to minimize data-transfer between nodes. 

* Training Dataset (2.2)
    * we have found that unfiltered or lightly filtered versions of Common Crawl tend to have lower quality than more curated datasets. 
    * Therefore, we took 3 steps to improve the average quality of our datasets:
    (1) we downloaded and filtered a version of CommonCrawl based on similarity to a range of high-quality reference corpora, 
    (2) we performed fuzzy deduplication at the document level, within and across datasets, to prevent redundancy and preserve the integrity of our held-out validation set as an accurate measure of overfitting, 
    (3) we also added known high-quality reference corpora to the training mix to augment CommonCrawl and increase its diversity.
        * an expanded version of the WebText dataset [RWC+19], 
        * two internet-based books corpora (Books1 and Books2) 
        * English-language Wikipedia.
    * CC total size: constituting 45TB of compressed plaintext before filtering and 570GB after filtering, roughly equivalent to 400 billion byte-pair-encoded tokens. 
    * during training, datasets are not sampled in proportion to their size, but rather datasets we view as higher-quality are sampled more frequently,
    * This essentially accepts a small amount of overfitting in exchange for higher quality training data.

* Training Process (2.3)
    * As found in [KMH+20, MKAT18], larger models can typically use a larger batch size, but require a smaller learning rate. 
    * We measure the gradient noise scale during training and use it to guide our choice of batch size [MKAT18]. 
    * To train the larger models without running out of memory, we use a mixture of model parallelism within each matrix multiply and model parallelism across the layers of the network. 
    * All models were trained on V100 GPU?s on part of a high-bandwidth cluster provided by Microsoft. 

* Evaluation (2.4)
    * For few-shot learning, we evaluate each example in the evaluation set by randomly drawing K examples from that task?s training set as conditioning, delimited by 1 or 2 newlines depending on the task. 
    * K can be any value from 0 to the maximum amount allowed by the model?s context window, which is nctx = 2048 for all models and typically fits 10 to 100 examples.
    *  Larger values of K are usually but not always better, 
    * For some tasks (see Appendix G) we also use a natural language prompt in addition to (or for K = 0, instead of) demonstrations.
    * On tasks that involve choosing one correct completion from several options (multiple choice), 
        * we provide K examples of context plus correct completion, 
        * followed by one example of context only, 
        * and compare the LM likelihood of each completion. 
    * For most tasks we compare the per-token likelihood (to normalize for length), 
    * however on a small number of datasets (ARC, OpenBookQA, and RACE) we gain additional benefit by normalizing by the unconditional probability of each completion: P(completion|context) / P(completion|answer context) , 
    * On tasks that involve binary classification, we give the options more semantically meaningful names (e.g. ?True? or ?False? rather than 0 or 1) and then treat the task like multiple choice
    * On tasks with free-form completion, we use beam search with the same parameters as [RSR+19]: 
        * a beam width of 4 and length penalty of alpha = 0:6. 
        * We score the model using F1 similarity score, BLEU, or exact match, depending on what is standard for the dataset at hand.
    * see Appendix G for details

* Results (3)
    * As observed in [KMH+20], language modeling performance follows a power-law when making efficient use of training compute. 
    * After extending this trend by two more orders of magnitude, we observe only a slight (if any) departure from the power-law. 
    * 9 groups
    * traditional language modeling tasks and tasks that are similar to language modeling, such as Cloze tasks and sentence/paragraph completion tasks. (3.1)
    * closed book? question answering tasks: tasks which require using the information stored in the model?s parameters to answer general knowledge questions. (3.2)
    *  model?s ability to translate between languages (especially one-shot and few-shot) (3.3)
    * model?s performance on Winograd Schema-like tasks (3.4)
    * datasets that involve commonsense reasoning or question answering (3.5)
    * reading comprehension tasks (3.6)
    * SuperGLUE benchmark suite (3.7)
    * NLI (3.8)
    * invent some additional tasks designed especially to probe in-context learning abilities (3.9)
    * We evaluate all tasks in the few-shot, one-shot, and zero-shot settings.

* Measuring and Preventing Memorization Of Benchmarks (4)
    * thoughts/analysis about contamination - training data leaking to the dev/test sets

* discussion (5)
    * On text synthesis, although the overall quality is high, GPT-3 samples still sometimes repeat themselves semantically at the document level, start to lose coherence over sufficiently long passages, contradict themselves, and occasionally contain non-sequitur sentences or paragraphs. 
    * Within the domain of discrete language tasks, we have noticed informally that GPT-3 seems to have special difficulty with ?common sense physics?, despite doing well on some datasets (such as PIQA [BZB+19])
    * Quantitatively, GPT-3?s in-context learning performance has some notable gaps on our suite of benchmarks, as described in Section 3, and in particular it does little better than chance when evaluated one-shot or even few-shot on some ?comparison? tasks, such as determining if two words are used the same way in a sentence, or if one sentence implies another (WIC and ANLI respectively), as well as on a subset of reading comprehension tasks. 
      This is especially striking given GPT-3?s strong few-shot performance on many other tasks.
    * We focused on exploring in-context learning behavior in autoregressive language models because it is straightforward to both sample and compute likelihoods with this model class. 
        * Thus our design decision comes at the cost of potentially worse performance on tasks which empirically benefit from bidirectionality. 
        * We also conjecture, based on past literature, that a large bidirectional model would be stronger at fine-tuning than GPT-3 (good future work)
    * the GPT-3 approach may eventually run into (or could already be running into) the limits of the pretraining objective. 
        * Our current objective weights every token equally and lacks a notion of what is most important to predict and what is less important. [
        * RRS20] demonstrate benefits of customizing prediction to entities of interest. 
        * useful language systems (for example virtual assistants) might be better thought of as taking goal-directed actions rather than just making predictions. 
    * Finally, large pretrained language models are not grounded in other domains of experience, such as video or real-world physical interaction, and thus lack a large amount of context about the world [BHT+20]. 
    * Another limitation broadly shared by language models is poor sample efficiency during pre-training. 
        * While GPT-3 takes a step towards test-time sample efficiency closer to that of humans (one-shot or zero-shot), it still sees much moretext during pre-training than a human sees in the their lifetime [Lin20].
    * expensive and inconvenient to perform inference on, which may present a challenge for practical applicability of models of this scale in their current form. 
        * One possible future direction to address this is distillation [HVD15] 
        * Distillation is well-explored in general [LHCG19a] but has not been tried at the scale of hundred of billions parameters;
    * GPT-3 shares some limitations common to most deep learning systems ? its decisions are not easily interpretable,
    * it is not necessarily well-calibrated in its predictions on novel inputs as observed by the much higher variance in performance than humans on standard benchmarks, 
    it retains the biases of the data it has been trained on. 

================================================================================
MetaICL: Learning to Learn In Context
https://arxiv.org/abs/2110.15943?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter
================================================================================

* Large language models (LMs) have recently been shown to be able to do in-context learning (Brown et al., 2020), 
    * where they learn a new task simply by conditioning on a few training examples and predicting which tokens best complete a test input.
    * is attractive because the model learns a new task through inference alone, without any parameter updates. 
    * However, performance significantly lags behind supervised finetuning, results are often high variance (Zhao et al., 2021; Perez et al., 2021), 

* In MetaICL, 
    * each meta-training example matches the test setup
    * it includes k + 1 training examples from one task that will be presented together as a single sequence to the language model, 
    * the output of the final example is used to calculate the cross-entropy training loss. 

* the model learns to recover the semantics of the task from the given examples, as must be done for in-context learning of a new task at test time. 
* MetaICL is distinct from other similar work as it allows learning new tasks from k examples alone, without relying on a task reformatting  (e.g., reducing everything to question answering) or task-specific templates (e.g., converting different tasks to a language modeling problem).
* 52 unique target tasks in total, which is the largest among all recent related work to the best of our knowledge.
* Gains over multi-task zero-shot transfer are particularly significant when meta-training tasks and target tasks are dissimilar, e.g. there are large differences in task formats, domains, or required skills. 
* we demonstrate MetaICL without any templates is better than recent work using human-written natural instructions, while the best performance is achieved by combining both approaches.

* MetaICL (3)
    * check table 1
    * The key idea is to use a multi-task learning scheme over a large collection of metatraining tasks
    * Following previous literature (Brown et al., 2020), the training examples are concatenated and provided as an single input to the model, which is feasible for k-shot learning (e.g., k = 16).
    * At test time, the model is evaluated on an unseen target task that comes with k training examples, and inference directly follows the same data format as in meta-training.

* Meta-training (3.1)
    * The model is meta-trained on a collection of tasks which we call meta-training tasks. 
    * For every iteration, one meta-training task is sampled, and k + 1 training examples (x1; y1); ...; (xk+1; yk+1) are sampled from the training examples of the chosen task. 
    * We then supervise the model by feeding the concatenation of x1; y1; ...; xk; yk; xk+1 to the model as an input 
    * and train the model to generate yk+1 using a negative log likelihood objective.
    * This simulates in-context learning at inference where the first k examples serve as training examples and the last (k + 1)-th example is regarded as the test example.

* Inference (3.2)
    * For a new target task, the model is given k training examples (x1; y1); ...; (xk; yk) as well as a test input x. 
    * It is also given a set of candidates C which is either a set of labels (in classification) or answer options (in question answering). 
    * As in meta-training, the model takes a concatenation of x1; y1; ...; xk; yk; x as the input, 
    * and compute the conditional probability of each label c_i \in C. 
    * The label with the maximum conditional probability is returned as a prediction.

* Channel MetaICL (3.3)
    * a noisy channel variant of MetaICL called Channel MetaICL, following Min et al. (2021). 
    * P(y|x) is reparameterized to P(x|y)P(y)/P(x)  that approximates/proportional P(x|y)P(y).
    * We follow Min et al. (2021) in using P(y) = 1
    * allows us to use the channel approach by simply flipping x and y. 
    * at meta-training time, the model is given a concatenation of y1; x1; ...; yk; xk; yk+1, and is trained to generate xk+1. 
    * At inference, the model computes argmax_c P(x|y1,x1; ...; yk,xk, c)

4 Experimental Setup

* Datasets (4.1)
    * We have 142 unique tasks in total - CROSSFIT (Ye et al., 2021) and UNIFIEDQA (Khashabi et al., 2020)
    * there is no overlap between the meta-training and target tasks. 
    * The number of unique target tasks in total is 52
    * We experiment with seven distinct settings as shown in Table 2,
        * HR!LR (High resource to low resource): 
            * We experiment with a main setting where datasets with 10,000 or more training examples are used as metatraining tasks and the rest are used as target tasks.
            * We think using high resource datasets for metatraining and low resource datasets as targets is a realistic and practical setting for few-shot learning.
        * X!X (X={Classification, QA}): 
            * We also experiment with two settings with meta-training and target tasks sharing the task format, although with no overlap in tasks.
        * Non-X!X (X={Classification, QA, NLI, Paraphase}):
            * four settings where meta-training tasks do not overlap with target tasks in task format and required capabilities.
            * These settings require the most challenging generalization capacities.
    * Each setting has a subset of target tasks with no domain overlap with any meta-training tasks (e.g., finance, poem, climate or medical). 

* Baselines (4.2)
    * We compare MetaICL and Channel MetaICL (table 3)
    * 0-shot: We use a pretrained LM as it is and run zero-shot inference, following Brown et al. (2020).
    * In-context: We use the pretrained LM as it is and use in-context learning by conditioning on a concatenation of k training examples, following Brown et al. (2020).
    * PMI 0-shot, PMI In-context: We use the PMI method from Holtzman et al. (2021); Zhao et al. (2021) for 0-shot and In-context learning.
    * Channel 0-shot, Channel In-context: We use the noisy channel model from Min et al. (2021) for 0-shot and In-context learning.
    * Multi-task 0-shot: We train the LM on metatraining tasks and use zero-shot transfer on a target task, as done in Khashabi et al. (2020); Zhong et al. (2021); Wei et al. (2021).
    * Channel Multi-task 0-shot: We have a noisy channel variant of Multi-task 0-shot. 
    * Oracle: We train the LM on a given target task. This is not directly comparable to other methods as parameter updates are required for every target task.
    * Oracle w/ meta-train: We train the LM on metatraining tasks first and then further finetuned on a target task. This is not directly comparable to other methods for the same reason as above.

* Evaluation (4.3)
    * Macro-F1 for classification (More suitable than accuracy for imbalanced classification)
    * Accuracy for non-classification tasks
    * For a target task, we use k = 16 training examples, sampled uniformly at random. 
    * We relax the assumption of perfect balance between labels on k training examples, following Min et al. (2021).
    * Because in-context learning is known to have high variance, we use 5 different sets of k training examples. 
    * We first compute the average and the worst-case performance over seeds for every target task, and then report the macro-average of them over all target tasks.

* Elimination of templates (table 4)
    * human-authored templates to transform the inputoutput pair to a natural language sentence require expensive manual effort (as 136 different templates are required for 136 tasks in this paper) and cause unstable model performance due to many different ways of writing (Mishra et al., 2021a). 
    * We eliminate templates, using the given input (or a concatenation of inputs if there are multiple) and label words provided in the original datasets. 

* Experiment Details (4.4)
    * As a base LM, we use GPT-2 Large (Radford et al., 2019) which consists of 770M parameters.
    * For baselines without meta-training (raw LMs), we also compare with GPT-J (Wang and Komatsuzaki, 2021),6B parameters, public available
    * For meta-training, 
        * we use up to 16,384 training examples per task. 
        * We use a batch size of 8, 
        * learning rate of 10^-5
        * a sequence length of 1024.
    * For the baselines with no in-context learning,
        * sequence length of 256. 
    * We train the model for 30; 000 steps. 
    * use an 8-bit approximation of an Adam optimizer 
    * mixed precision 
    * Training was done for 4.5 hours with eight 32GB GPUs
    * This is drastically more efficient than recent prior work, e.g., 270 hours of a 512GB TPU in Sanh et al. (2021).

* Experimental Results (5)
    * Our baselines are strong     
        * Among raw LMs without meta-training, we observe that channel in-context baselines are the most competitive, consistent with findings from Min et al. (2021). 
        * Multi-task 0-shot baselines do not outperform the best raw LM baseline in most settings, despite being supervised on a large set of meta-training tasks (this somewhat contradicts findings from Wei et al. (2021); Sanh et al. (2021))
            * our models are much smaller than theirs (770M vs. 11B-137B);
            * in fact, Wei et al. (2021) reports Multi-task 0-shot starts to be better than raw LMs only when the model size is 68B or larger. 
            * we compare with much stronger channel baselines which they did not; 
        * Multi-task 0-shot outperforms non-channel LM baselines but not channel LM baselines.
    * MetaICL outperforms baselines 
        * While which of MetaICL or Channel MetaICL is better depends on the setting, 
        * gains over baselines in the HR!LR, non-NLI!NLI and non-Para!Para settings are significant. 
            * This is intriguing because HR!LR is the most realistic setting, and the other  two settings are those in which target tasks require very different skills from meta-training tasks. 
            * This demonstrates that MetaICL enables the model to recover the semantics of the task in context at inference even though there is no similar tasks seen at training time.
        * Channel MetaICL generally achieves good performance, except in the QA!QA setting. 
            * likely because meta-training and target tasks are all relatively similar, so it does not require significant generalization capacity and Multi-task 0-shot baseline achieves very strong performance. 
    * Gains are larger on unseen domains 
    * Comparison to oracle 
        * MetaICL matches or sometimes even outperforms performance of oracle without meta-training. 
        * This is a promising signal, given that no prior work has shown models with no parameter updates on the target can match or outperform supervised models. 
        * oracle with meta-training outperforms oracle without meta-training?so meta-training also helps in supervised learning?as well as MetaICL. 
        * This hints that there is still room for improvement in methods that allow learning without parameter updates .
    * Comparison to GPT-J 
        * MetaICL, despite being 8x smaller, outperforms or matches GPT-J baselines, with an exception in QA for which GPT-J achieves particularly strong performance.
    * We also note that GPT-J is not much better than GPT-2 Large when compared within raw LM baselines  except for QA. 
        * In unseen domains, however, GPT-J is consistently better. 
        * This is likely due to differences in the pretraining data 

* Ablations (5.2)
    * Number of meta-training tasks 
        * On average, performance generally increases as the number of tasks increase, which is consistent with results in Mishra et al. (2021b); Wei et al. (2021). 
        * Across different numbers of meta-training tasks, Channel MetaICL consistently outperforms other models. 
        * the choice of meta-training tasks gives substantial impact in performance.
    * Diversity in meta-training tasks 
        * MetaICL with a diverse set outperforms MetaICL with a non-diverse set by a substantial margin. 
        * We think that diversity among meta-training tasks is one of substantial factors that impact the success of MetaICL, although likely not the only factor.
            * choice of meta-training tasks, such as (1) high quality data with diverse domains tend to help and (2) adversarially collected data tends to be unhelpful.  
    * Are instructions necessary? 
        * On one hand, learning to condition on k examples may remove the necessity of instructions. On the other hand, instructions may still be complementary and provide the model with extra useful information.
        * To summarize,
            (1) learning to in-context learn (MetaICL) outperforms learning to learn from instructions; 
            (2) MetaICL and using instructions are largely complementary,
            (3) MetaICL actually benefits more from using instructions than Multi-task 0-shot does.

================================================================================
Finetuned Language Models Are Zero-Shot Learners
https://arxiv.org/abs/2109.01652?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter
================================================================================

================================================================================
ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning
https://arxiv.org/abs/2111.10952
================================================================================

================================================================================
Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)
https://arxiv.org/abs/2110.08207?utm_campaign=NLP%20News&utm_medium=email&utm_source=Revue%20newsletter
================================================================================

* Can zero-shot generalization instead be directly induced by explicit multitask learning? 
* To test this question at scale, we develop a system for easily mapping general natural language tasks into a human-readable prompted form. 
* We convert a large set of supervised datasets, each with multiple prompts using varying natural language. 
* We fine-tune a pretrained encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture 
* The model attains strong zero-shot performance on several standard datasets, often outperforming models up to 16x its size. and huggingface.co/bigscience/T0pp.

* An influential hypothesis is that large language models generalize to new tasks as a result of an implicit process of multitask learning (Radford et al., 2019). 
* As a byproduct of learning to predict the next word, a language model is forced to learn from a mixture of implicit tasks included in their pretraining corpus. 
* For example, by training on generic text from a web forum, a model might implicitly learn the format and structure of question answering. 
* However, this ability requires a sufficiently large model and is sensitive to the wording of its prompts (Perez et al., 2021; Zhao et al., 2021; Reynolds and McDonell, 2021).
* Yet, it is an open question how implicit this multitask learning really is. 
* Given the scale of large language models and the datasets they are trained on, this explicit multitask supervision could feasibly play a large role in zero-shot generalization.

* In this paper, we instead focus on intentionally and explicitly training large language models in a supervised and massively multitask fashion. 
    * Our approach uses a multitask training mixture made up of a large set of different tasks specified in natural language prompts. 
    * Our goal is to induce a model to better generalize to unseen tasks without requiring massive scale, as well as being more robust to the wording choices of the prompts. 
    * To convert a large set of natural language tasks into prompted form, we use a simple templating language for structured datasets. 
    * We develop an interface for prompt collection from public contributors that facilitated the collection of a large multitask mixture with multiple prompts per dataset.
    * We then train a variant of the T5 encoder-decoder model (Raffel et al., 2020; Lester et al., 2021) 

* Our experiments study two questions. 
    * First, does multitask prompted training improve generalization to unseen tasks? 
        * Yes. By showing that our model matches or exceeds the performance of GPT-3 (Brown et al., 2020) on 9 out of 11 held-out datasets, despite being about 16 smaller. 
    * Second, does training on a wider range of prompts improve robustness to prompt wording? 
        * we find that training on more prompts per dataset consistently improves the median and decreases the variability of performance on held-out tasks. 
        * Training on prompts from a wider range of datasets also generally improves the median but does not decrease the variability.

* In this work, we distinguish implicit multitask learning in language model pretraining from explicit multitask learning (Caruana, 1997), the technique for mixing multiple tasks into a single supervised training process.
* Natural language prompting is the method of reformatting NLP tasks in the format of a natural language response to natural language input. 

* Finally, in explaining the success of prompts, the leading hypothesis is that models learn to understand the prompts as task instructions which help them generalize to unseen tasks (Wei et al., 2021; Mishra et al., 2021; Schick and Schutze, 2021; Brown et al., 2020). 
* However, the extent to which this success depends on the semantic meaningfulness of the prompts has been challenged (Webson and Pavlick, 2021; Logan et al., 2021). 
* Thus, in this work, we remain agnostic as to why prompts support generalization. We only claim that prompts serve as a natural format for multitask training which empirically supports generalization to unseen tasks.

* MEASURING GENERALIZATION TO UNSEEN TASKS (3)
    * We use the term ?task? to refer to a general NLP ability that is tested by a group of specific datasets. 
    * To evaluate zero-shot generalization to new tasks, we train on a subset of tasks and evaluate on a held-out group of tasks.
    * Noting that grouping by task is an imperfect heuristic, we err on the side of organizing our task taxonomy based on the task format as opposed to required skill, largely based on conventions in the literature (Khashabi et al., 2020b; Vu et al., 2020; Ye et al., 2021). 
    * All experiments use datasets in the Hugging Face datasets library (Lhoest et al., 2021).
    * To test zero-shot generalization, we hold out all constituent datasets of four tasks: natural language inference (NLI), sentence completion, word sense disambiguation, and coreference resolution. 
    * Additionally, we do not train our main model on any datasets that GPT- 3 used for evaluation, so that our main results will be a fair zero-shot comparison. 
    * We verify that data for those tasks is not leaked through the pretraining corpus as detailed in Appendix E. 
    * *we also evaluate on a subset of the datasets from BIG-Bench (BIG-bench collaboration, 2021),

* A UNIFIED PROMPT FORMAT (4)
    * All datasets are given to our model in natural language prompted form to enable zero-shot experimentation.
    * To facilitate writing a large collection of prompts, we develop a templating language and an application that make it easy to convert diverse datasets into prompts. 
    *  We define a prompt asconsisting of an input template and target template, along with a collection of associated metadata.
    * The templates are functions mapping a data example into natural language for the input and target sequences. 
    * * For example, in the case of an NLI dataset, 
        * the example would include fields for Premise, Hypothesis, Label. 
        An input template whereas a target template can be defined with the label choices (Choices[label])
        * Here Choices is prompt-specific metadata that consists of the options yes, maybe, no corresponding to label being entailment (0), neutral (1) or contradiction (2). 
    * We refer to this collection as the Public Pool of Prompts (P3). 
        * P3 contains 1939 prompts for 171 datasets (11.3 prompts per dataset on average). 
        * These prompts contain on average 14.4 tokens, not including variables and other elements from the templating language.  
    * All prompts used in our experiments are sourced from P3 (except for BIG-Bench, for which the prompts are provided by its maintainers).

* Model (5.1)
    * we fine-tune a pretrained model on our multi-task training mixture of natural language prompted datasets. 
    * Our model uses an encoder-decoder architecture with input text fed to the encoder and target text produced by the decoder. 
    * The model is trained to autoregressively generate the target through standard maximum likelihood training. 
    * Unlike decoder-only language models such as GPT-3, it is never trained to generate the input.
    * All models we trained are based on T5, a Transformer-based encoder-decoder language model pretrained with a masked language modeling-style objective on 1T tokens from C4 (Raffel et al., 2020).
    * Since T5?s pretraining objective involves filling in tokens from the input text that have been removed,  it is quite different from the conditional text generation format used in our prompted datasets. 
    * we use T5+LM from Lester et al. (2021), which was produced by training T5 on 100B additional tokens from C4 on a standard language modeling objective. 
    * Unless specified otherwise, we use the XXL version which has 11B parameters.

* TRAINING (5.2)
    * T0: trained on the multitask mixture detailed in Section 3 (i.e. yellow datasets in Figure 2). 
    * T0+ is the same model but trained on a mixture that adds GPT-3?s evaluation datasets. 
    * For T0++, we add GPT-3?s and SuperGLUE (Wang et al., 2019a)?s datasets to the training mixture which includes some held-out tasks. 
    * We also consider T0 (3B), which corresponds to the smaller 3 billion-parameter XL variant of T5
    * We perform checkpoint selection by choosing the checkpoint that yields the highest score on the validation splits of our training datasets. 
    * This still satisfies true zero-shot (Perez et al., 2021) setting as we do not use any examples from any of the held-out tasks to select the best checkpoint. 
    * At a high level, we assemble our multitask training mixture simply by combining all of the examples from all training datasets and shuffling the result. 
    * This is equivalent to sampling from each dataset in proportion to the number of examples in the dataset.  
    * However, the number of examples in each of our training datasets varies by two orders of magnitude. 
    * We therefore follow the strategy used in Raffel et al. (2020) and treat any dataset with over 500?000 examples as having 500?000 / num templates examples for the purposes of sampling, where num templates is the number of templates created for the dataset. 
    * We feed the model input and target sequences of 1024 and 256 tokens, respectively. 
    * Following Raffel et al. (2020), we use packing to combine multiple training examples into a single sequence to reach the maximum sequence length.
    * We use a batch size of 1024 sequences (corresponding to 220 total input tokens per batch) 
    * the Adafactor optimizer (Shazeer and Stern, 2018). 
    * *Following standard practice for fine-tuning T5, we use a learning rate of 1e-3 and a dropout rate of 0.1.

* EVALUATION (5.3)
    * We evaluate zero-shot generalization on 11 NLP datasets in 4 unseen tasks: natural language inference, coreference, word sense disambiguation, and sentence completion (green datasets in Figure 2). 
    * Unless specified otherwise, we report numbers on the validation splits. 
    * We also evaluate on 14 datasets from BIG-Bench (BIG-bench collaboration, 2021).
    * For tasks that involve choosing the correct completion from several options (e.g. multiple choice), we follow Brown et al. (2020) and use rank scoring to evaluate our model: 
    * we compute the log- likelihood of each of the target options under the fine-tuned model and select the option with the highest log-likelihood as the prediction. 
    * For simplicity, we do not apply any normalization strategies to the log-likelihoods and use them as they are. 
    * We report accuracy for every dataset.
    * We do not perform prompt selection by comparing the performance of different prompts on the validation split; 
        * Perez et al. (2021) highlights how such a strategy leaks information from the evaluation splits, which makes the evaluation not ?true? zero-shot. 
    * For a given dataset, we report the median performance across the prompts for this dataset (up to 15) along with their interquartile range (Q3 - Q1) to measure the sensitivity of the model to the wording of the prompts.

* RESULTS (6)

* GENERALIZATION TO UNSEEN TASKS (6.1)
    * In Figure 4, we compare T0 against our T5+LM baseline on four held-out tasks: natural language inference, coreference, sentence completion, and word sense disambiguation. 
    * Our approach leads to significant gains over our baseline on all datasets, indicating the benefits of multitask training compared to only language modeling with an identical model and prompts.
    * we compare our results to the zeroshot performance of various GPT-3 model variants up to 175B parameters. 
    * T0 surpasses the performance of all GPT-3 models on 8 out of 11 held-out datasets. 
    * Neither T0 nor GPT-3 were trained on natural language inference
    * T0 outperforms GPT-3 on all NLI datasets (even though T5+LM does not). 
    * T0 underperforms GPT-3 significantly on HellaSwag, and Winogrande, as does the T5+LM baseline. 
        * We note though that for Winogrande, GPT-3 uses a specialized task format and evaluation procedure; we have not investigated whether these techniques would improve the performance of T0 or the baselines.
    * we assess the zero-shot performance of T0, T0+, and T0++ on a subset of the BIG-Bench benchmark (BIG-bench collaboration, 2021). 
        * BIG-Bench datasets come with their own prompts, prepared through a different process than P3. 
        * Tasks from BIG-Bench focus on a variety of skills not covered by our training tasks
        * We compare our model to a series of preliminary diagnostic baseline models trained by Google and evaluated by the BIG-Bench maintainers. 
        * These models are decoder-only Transformer language models trained on a standard language modeling objective with varying model size. 
        * We find that at least one of the T0 variants outperform all baseline models on all tasks except for StrategyQA. 
        * In most cases, the performance of our models improves as the number of training datasets increases (i.e. T0++ outperforms T0+ which outperforms T0).

* PROMPT ROBUSTNESS (6.2)
    * Our second research question is whether training on a wider range of prompts improves robustness to the wording of the prompts. 
    * Effect of More Prompts per Dataset 
        * we fix d and compare three models where
            * p = 1 (one randomly chosen original-task prompt per dataset), 
            * p = all available prompts (corresponding to T0, on average p = 8.03), 
            * p = 0 (corresponding to T5+LM without any prompted training).
        * We train all models with the same hyperparameters and the same number of steps. 
        * Figure 6 shows that, even with just one prompt per dataset (red), performance on unseen tasks can improve substantially over the baseline (blue)
        * although the spread (interquartile range between Q1 and Q3) does not appreciably improve with p = 1. 
        * However, further increasing p from 1 to an average of 8.03 does yield additional improvement in both median (increases for 11/11 datasets) and spread (decreases for 7/11 datasets). 
        * This reinforces our hypothesis that training on more prompts per dataset leads to better and more robust generalization to unseen tasks.
    * Effect of Prompts from More Datasets 
        * In this experiment, we fix p = all available prompts and increase d from 39 to 49 to 55 (T0, T0+, T0++, respectively, datasets are given in Section 5) increasing the total number of prompts seen during training. 
        * Figure 7 shows that the median performance of all 5 held-out datasets increases as d increases from 39 to 49. 
        * However, the spread only decreases for 1 out of 5 datasets. 
            * For some datasets (e.g., ANLI), this is an artifact of the fact that some prompts always perform poorly, so that when other prompts improve, the spread is stretched larger. 
            * For other datasets (e.g., CB), however, the spread does decrease in T0+. 
        * As d increases from 49 to 55, the median performance of all datasets again increases, but the spread only decreases for 2 out of 5 datasets. 
        * Although further investigation is needed, it appears that increasing d does not consistently make the model more robust to the wording of prompts.
    * Comparing T0 and GPT-3 robustness 
        * Because Brown et al. (2020) only report one prompt per dataset with no standard deviation, we evaluate GPT-3 on RTE using the 10 prompts we prepared through OpenAI?s API
        * results suggest that T0 is more robust to prompt formulation than GPT-3. - check numbers in the paper

* DISCUSSION OF SIMILAR APPROACHES (7)
    * Our results demonstrate that explicit multitask prompted fine-tuning substantially improves zeroshot generalization to unseen tasks
    * we discuss two other works that share a similar approach:
    * As details of the OpenAI model have not been published, and it is only available through a commercial API, we do not compare to it in our paper.
    * FLAN: Wei et al. (2021)
        * which largely follows the same method of enabling zero-shot generalization through multitask prompted training. 
        * They focus on fine-tuning standard autoregressive language models on datasets from a diverse collection of tasks and evaluating performance on a single held-out task at a time. 
        * Compared to FLAN, T0 zero-shot performance better on CB and RTE, similar on Story Cloze and COPA, and worse on Winogrande, ANLI, and HellaSwag. 
        * T0++ outperforms FLAN on CB, RTE, and COPA and matches FLAN?s performance on Winogrande and ANLI. 
        * Notably, T0 and T0++ attain this performance despite being over 10x smaller than FLAN (137B vs. 11B parameters). 
        * Surprisingly, Wei et al. (2021) perform an ablation with a model of comparable size (8B parameters) to T0 (11B parameters) and find that that performance on held-out tasks decreases after multi-task training.
        * key differences
            * We use an encoder-decoder model that was pretrained with a different objective (masked language modeling) before being trained as a standard language model and finally finetuned on the multitask mixture. 
              MLM  has repeatedly been shown to be a dramatically more effective pre-training strategy (Raffel et al., 2020; Baevski et al., 2019; Devlin et al., 2019).
            * Our prompts are qualitatively more diverse in terms of their length and creativity
            * We hold out multiple tasks at once, rather than only holding out a single task at a time. We made this choice in order to evaluate a single model?s ability to generalize to multiple diverse tasks.

================================================================================
ERNIE: Enhanced Language Representation with Informative Entities
https://arxiv.org/abs/1905.07129
================================================================================

* the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs),
* we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously.

* figure 1

* For incorporating external knowledge into language representation models, there are two main challenges.
    (1) Structured Knowledge Encoding: regarding to the given text, how to effectively extract and encode its related informative facts in KGs for language representation models is an important problem; 
    (2) Heterogeneous Information Fusion: the pre-training procedure for language representation is quite different from the knowledge representation procedure, leading to two individual vector spaces. How to design a special pre-training objective to fuse lexical, syntactic, and knowledge information?

* we propose Enhanced Language RepresentatioN with Informative Entities (ERNIE),
    * pretrains a language representation model on both large-scale textual corpora and KGs:
    (1) For extracting and encoding knowledge information,
        * we firstly recognize named entity mentions in text and then align these mentions to their corresponding entities in KGs.
        * Instead of directly using the graph-based facts in KGs, we encode the graph structure of KGs with knowledge embedding algorithms like TransE (Bordes et al., 2013),
        * and then take the informative entity embeddings as input for ERNIE.
        * Based on the alignments between text and KGs, ERNIE integrates entity representations in the knowledge module into the underlying layers of the semantic module.
    (2) Similar to BERT, we adopt the masked language model and the next sentence prediction as the pre-training objectives.
        * for the better fusion of textual and knowledge features, we design a new pre-training objective by randomly masking some of the named entity alignments in the input text and asking the model to select appropriate entities from KGs to complete the alignments.
        * our objectives require models to aggregate both context and knowledge facts for predicting both tokens and entities, and lead to a knowledgeable language representation model.

* We conduct experiments on two knowledgedriven NLP tasks,
    * i.e.,  entity typing and relation classification.
    * better than BERT

3 Methodology

3.1 Notations

* n is the length of the token sequence.
* m is the length of the entity sequence. 
* Note that m is not equal to n in most cases, as not every token can be aligned to an entity in KGs. 
* we denote the whole vocabulary containing all tokens as V
* and the entity list containing all entities in KGs as E. 
* If a token w has a corresponding entity e, their alignment is defined as f(w) = e.
* In this paper, we align an entity to the first token in its named entity phrase, as shown in Figure 2.

3.2 Model Architecture

* As shown in Figure 2, the whole model architecture of ERNIE consists of two stacked modules:
    (1) the underlying textual encoder (T-Encoder) responsible to capture basic lexical and syntactic information from the input tokens,
    (2) the upper knowledgeable encoder (K-Encoder) responsible to integrate extra token-oriented knowledge information into textual information from the underlying layer, so that we can represent heterogeneous information of tokens and entities into a united feature space.
* we denote the number of T-Encoder layers as N,
* the number of K-Encoder layers as M.
* In this paper, tokens are at the subword level.

* To be specific, 
    * given a token sequence w_1, ..., w_n
    * and its corresponding entity sequence e_1, ..., e_m
    * the textual encoder firstly sums the token embedding, segment embedding, positional embedding for each token to compute its input embedding, 
    * and then computes lexical and syntactic features w1, ..., w_n
    *  {w_1, , w_n} = T-Encoder({w_1, ..., w_n})
    * T-Encoder is like BERT

* After computing {w_1, ..., w_n}, ERNIE adopts a knowledgeable encoder K-Encoder to inject the knowledge information into language representation.
* To be specific,
    * we represent {e_1, ..., e_m} with their entity embeddings {e_1, ..., e_m}, which are pre-trained by the effective knowledge embedding model TransE (Bordes et al., 2013).
    * Then, both {w_1, ..., w_n} and {e_1, ..., e_m} are fed into K-Encoder for fusing heterogeneous information and computing final output embeddings,
    * {w_1^o, ..., w_n^o},{e_1^o, ..., e_m^o} = K-Encoder({w_1, ..., w_n},{e_1, ..., e_m})
    * o: means features for a specific task

3.3 Knowledgeable Encoder

* designed for encoding both tokens and entities as well as fusing their heterogeneous features.
* In the i-th aggregator, the input token embeddings w and entity embeddings e, from the preceding aggregator are fed into two multi-head self-attentions (MH-ATTs) respectively,

{w_1^i, ..., w_n^i} = MH-ATT({w_1^i-1, ..., w_n^i-1})
{e_1^i, ..., e_m^i} = MH-ATT({e_1^i-1, ..., e_m^i-1})

* Then, the i-th aggregator adopts an information fusion layer for the mutual integration of the token and entity sequence

h_j = GELU(W_t^i * w_j^i + W_e^i * e_k^i + b^i)
w_j^i = GELU(W_t^i * h_j + b_t^i)
e_k^i = GELU(W_e^i * h_j + b_e^i)

* For the tokens without corresponding entities, the information fusion layer computes the output embeddings without integration as follows,

h_j = GELU(W_t^i * w_j^i + b^i)
w_j^i = GELU(W_t^i * h_j + b_t^i)

* For simplicity, the i-th aggregator operation is denoted as follows,

{w_1^i, ..., w_n^i},{e_1^i, ..., e_m^i} = Aggregator({w_1^i-1, ..., w_n^i-1},{e_1^i-1, ..., e_m^i-1})

* The output embeddings of both tokens and entities computed by the top aggregator will be used as the final output embeddings of the knowledgeable encoder K-Encoder.

3.4 Pre-training for Injecting Knowledge

* we propose a new pre-training task for ERNIE, which randomly masks some token-entity alignments and then requires the system to predict all corresponding entities based on aligned tokens.
* As our task is similar to training a denoising auto-encoder (Vincent et al.,2008), we refer to this procedure as a denoising entity auto-encoder (dEA).
* Considering that the size of E is quite large for the softmax layer, we thus only require the system to predict entities based on the given entity sequence instead of all entities in KGs.
* we define the aligned entity distribution for the token wi as follows,
    * check formula - softmax with w and e

* Considering that there are some errors in tokenentity alignments, we perform the following operations for dEA:
    (1) In 5% of the time, for a given token-entity alignment, we replace the entity with another random entity, which aims to train our model to correct the errors that the token is aligned with a wrong entity; 
    (2) In 15% of the time, we mask token-entity alignments, which aims to train our model to correct the errors that the entity alignment system does not extract all existing alignments; 
    (3) In the rest of the time, we keep tokenentity alignments unchanged, which aims to encourage our model to integrate the entity information into token representations for better language understanding.

* Similar to BERT, ERNIE also adopts the masked language model (MLM) and the next sentence prediction (NSP) 
    * The overall pre-training loss is the sum of the dEA, MLM and NSP loss.

3.5 Fine-tuning for Specific Tasks

* As shown in Figure 3, for various common NLP tasks, ERNIE can adopt the fine-tuning procedure similar to BERT.
* We can take the final output embedding of the first token, which corresponds to the special [CLS] token, as the representation of the input sequence for specific tasks.
* For some knowledge-driven tasks (e.g., relation classification and entity typing), we design special finetuning procedure:
* For relation classification,
    * modifies the input token sequence by adding two mark tokens to highlight entity mentions.
    * These extra mark tokens play a similar role like position embeddings in the conventional relation classification models (Zeng et al., 2015).
    * Then, we also take the [CLS] token embedding for classification.
    * Note that we design different tokens [HD] and [TL] for head entities and tail entities respectively.
* The specific fine-tuning procedure for entity typing 
    * is a simplified version of relation classification.
    * we argue that the modified input sequence with the mention mark token [ENT] can guide ERNIE to combine both context information and entity mention information attentively.

4 Experiments

4.1 Pre-training Dataset

* we adopt the parameters of BERT released by Google3 to initialize the Transformer blocks for encoding tokens.
* Since pre- training is a multi-task procedure consisting of NSP, MLM, and dEA,
    * we use English Wikipedia as our pre-training corpus and align text to Wikidata.
* After converting the corpus into the formatted data for pre-training,
    * the annotated input has nearly 4, 500M subwords and 140M entities,
    * and discards the sentences having less than 3 entities.
* Before pre-training ERNIE
    * we adopt the knowledge embeddings trained on Wikidata4 by TransE as the input embeddings for entities.
    * To be specific,
        * we sample part of Wikidata which contains 5, 040, 986 entities and 24, 267, 796 fact triples.
        * The entity embeddings are fixed during training and the parameters of the entity encoding modules are all initialized randomly.

4.2 Parameter Settings and Training Details
* we denote the hidden dimension of token embeddings and entity embeddings as H_w, H_e respectively
* and the number of self-attention heads as A_w, A_e respectively.
* we have the following model size:
    * N = 6, M = 6, H_w = 768, H_e = 100, A_w = 12, A_e = 4.
* The total parameters are about 114M.
    * The total amount of parameters of BERTBASE is about 110M,
    * which means the knowledgeable module of ERNIE is much smaller than the language module and has little impact on the run-time performance.
* we only pre-train ERNIE on the annotated corpus for one epoch.
* To accelerate the training process, we reduce the max sequence length from 512 to 256 as the computation of selfattention is a quadratic function of the length.
* To keep the number of tokens in a batch as same as BERT, we double the batch size to 512.
Except for setting the learning rate as 5e-5, we largely follow the pre-training hyper-parameters used in BERT.

* For fine-tuning,
    * most hyper-parameters are the same as pre-training, except batch size, learning rate, and number of training epochs.
    * batch size:32, 
    * learning rate (Adam):5e-5, 3e-5, 2e-5, 
    * number of epochs ranging from 3 to 10.

* We also evaluate ERNIE on the distantly supervised dataset, i.e., FIGER (Ling et al., 2015).
* As the powerful expression ability of deeply stacked Transformer blocks, we found small batch size would lead the model to overfit the training data.
    * Hence, we use a larger batch size and less train- ing epochs to avoid overfitting,
    * and keep the range of learning rate unchanged, i.e., batch size:2048, number of epochs:2,3.

* As most datasets do not have entity annotations, we use TAGME (Ferragina and Scaiella, 2010) to extract the entity mentions in the sentences and link them to their corresponding entities in KGs.

4.3 Entity Typing
* Given an entity mention and its context, entity typing requires systems to label the entity mention with its respective semantic types.

* To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets 
    * FIGER (Ling et al., 2015) 
        * The training set of FIGER is labeled with distant supervision, and its test set is annotated by human.
    * Open Entity (Choi et al., 2018).
        * completely manually-annotated dataset.
    * The statistics of these two datasets are shown in Table 1.

* We compare our model with the following baseline models for entity typing:
    * NFGEC.
        * a hybrid model proposed by Shimaoka et al. (2016).
        * it combines the representations of entity mention, context and extra hand-craft features as input, and is the stateof- the-art model on FIGER.
        * As this paper focuses on comparing the general language representation abilities of various neural models, we thus do not use the hand-craft features in this work.
    * UFET.
        * For Open Entity, we add a new hybrid model UFET (Choi et al., 2018) for comparison.
        * UFET is proposed with the Open Entity dataset, which uses a Bi-LSTM for context representation instead of two Bi-LSTMs separated by entity mentions in NFGEC.

* Besides NFGEC and UFET, we also report the result of fine-tuning BERT with the same input format introduced in Section 3.5 for fair com parison.
* we compare NFGEC, BERT, ERNIE on FIGER,
    * adopt strict accuracy, loose macro, loose micro scores for evaluation.
* We compare NFGEC, BERT, UFET, ERNIE on Open Entity,
    * adopt precision, recall, micro- F1 scores for evaluation.

* The results on FIGER are shown in Table 2.
    (1) * BERT achieves comparable results with NFGEC on the macro and micro metrics.
        * However, BERT has lower accuracy than the best NFGEC model.
        * As strict accuracy is the ratio of instances whose predictions are identical to human annotations, it illustrates some wrong labels from distant supervision are learned by BERT due to its powerful fitting ability.
    (2) Compared with BERT, ERNIE significantly improves the strict accuracy, indicating the external knowledge regularizes ERNIE to avoid fitting the noisy labels and accordingly benefits entity typing.
* The results on Open Entity are shown in Table 3.
    (1) BERT and ERNIE achieve much higher recall scores than the previous entity typing models, which means pre-training language models make full use of both the unsupervised pre-training and manuallyannotated training data for better entity typing.
    (2) Compared to BERT, ERNIE improves the precision by 2% and the recall by 2%, which means the informative entities help ERNIE predict the labels more precisely.

* In summary, ERNIE effectively reduces the noisy label challenge in FIGER, which is a widely-used distantly supervised entity typing dataset, by injecting the information from KGs.
* Besides, ERNIE also outperforms the baselines on Open Entity which has gold annotations.

4.4 Relation Classification

* To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets 
    * FewRel (Han et al., 2018c) 
        * As the original experimental setting of FewRel is few-shot learning, we rearrange the FewRel dataset for the common relation classification setting. 
        * we sample 100 instances from each class for the training set,
        * and sample 200 instances for the development and test respectively.
        * There are 80 classes in FewRel,
    * TACRED (Zhang et al., 2017).
        * there are 42 classes (including a special relation ?no relation?) in TACRED.
* The statistics of these two datasets are shown in Table 4.

* We compare our model with the following baseline models for relation classification:
    * CNN. With a convolution layer, a max-pooling layer, and a non-linear activation layer,
        * CNN gets the output sentence embedding, and then feeds it into a relation classifier.
        * To better capture the position of head and tail entities, position embeddings are introduced into CNN (Zeng et al., 2015; Lin et al., 2016; Wu et al., 2017; Han et al., 2018b).
    * PA-LSTM. Zhang et al. (2017) 
        * introducing a position-aware attention mechanism over an LSTM network,
        * which evaluates the relative contribution of each word in the sequence for the final sentence representation.
    * C-GCN. Zhang et al. (2018) 
        * adopt the graph convolution operations to model dependency trees for relation classification.
        * To encode the word order and reduce the side effect of errors in dependency parsing, Contextualized GCN (C-GCN) firstly uses Bi-LSTM to generate contextualized representations as input for GCN models.
* In addition to these three baselines, we also finetune BERT with the same input format introduced in Section 3.5 for fair comparison.

* As FewRel does not have any null instance where there is not any relation between entities, we adopt macro averaged metrics to present the model performances.
* Since FewRel is built by checking whether the sentences contain facts in Wikidata, we drop the related facts in KGs before pre-training for fair comparison.

* From Table 5, we have two observations:
    (1) As the training data does not have enough instances to train the CNN encoder from scratch, CNN just achieves an F1 score of 69:35%.
        However, the pre-training models including BERT and ERNIE increase the F1 score by at least 15%.
    (2) ERNIE achieves an absolute F1 increase of 3:4% over BERT,
        which means fusing external knowledge is very effective.

* In TACRED, there are nearly 80% null instances so that we follow the previous work (Zhang et al., 2017) to adopt micro averaged metrics to represent the model performances instead of the macro.
* The results of CNN, PA-LSTM, and C-GCN come from the paper by Zhang et al. (2018), which are the best results of CNN, RNN, and GCN respectively.
* From Table 5, we observe that:
    (1) The C-GCN model outperforms the strong BERT model by an F1 increase of 0:4%, as C-GCN utilizes the dependency trees and the entity mask strategy.
        The entity mask strategy refers to replacing each subject (and object similarly) entity with a special NER token, which is similar to our proposed pre-training task dEA.
    (2) ERNIE achieves the best recall and F1 scores, and increases the F1 of BERT by nearly 2:0%,
        which proves the effectiveness of the knowledgeable module for relation classification.

* In conclusion, we find that the pre-trained language models can provide more information for relation classification than the vanilla encoder CNN and RNN.
* And ERNIE outperforms BERT on both of the relation classification datasets, especially on the FewRel which has a much smaller training set. 
* It demonstrates extra knowledge helps the model make full use of small training data, which is important for most NLP tasks as large-scale annotated data is unavailable.

4.5 GLUE
* the main benchmark used in Devlin et al. (2019).
* we evaluate ERNIE on 8 datasets of GLUE and compare it with BERT.
* In Table 6,
    * we report the results of our evaluation submissions and those of BERT from the leaderboard.
    * We notice that ERNIE is consistent with BERTBASE on big datasets like MNLI, QQP, QNLI, and SST-2.
    * The results become more unstable on small datasets,
        * ERNIE is better on CoLA and RTE, 
        * but worse on STS-B and MRPC.
    * In short,     ERNIE achieves comparable results with BERTBASE on GLUE.
    * On the one hand, it means GLUE does not require external knowledge for language representation.
    * On the other hand, it illustrates ERNIE does not lose the textual information after heterogeneous information fusion.

4.6 Ablation Study
* we explore the effects of the informative entities and the knowledgeable pretraining task (dEA) for ERNIE using FewRel dataset.
* w/o entities and w/o dEA refer to finetuning ERNIE without entity sequence input and the pre-training task dEA respectively.
* As shown in Table 7,
    (1) Without entity sequence input, dEA still injects knowledge information into language representation during pre-training, which increases the F1 score of BERT by 0:9%.
    (2) Although the informative entities bring much knowledge informa tion which intuitively benefits relation classification, ERNIE without dEA takes little advantage of this, leading to the F1 increase of 0:7%.

5 Conclusion
* we propose ERNIE to incorporate knowledge information into language representation models.
* Accordingly, we propose the knowledgeable aggregator and the pre-training task dEA for better fusion of heterogeneous information from both text and KGs.
* The experimental results demonstrate that ERNIE has better abilities of both denoising distantly supervised data and fine-tuning on limited data than BERT.
* There are three important directions remain for future research:
    (1) inject knowledge into feature-based pre-training models such as ELMo (Peters et al., 2018); 
    (2) introduce diverse structured knowledge into language representation models such as ConceptNet (Speer and Havasi, 2012) which is different from the world knowledge database Wikidata; 
    (3) annotate more real-world corpora heuristically for building larger pre-training data.

================================================================================
ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation
https://arxiv.org/abs/2107.02137
================================================================================

* contributions
    * It fuses auto-regressive network and auto-encoding network,  so that the trained model can be easily tailored for both natural language understanding and generation tasks with zero-shot learning,  few-shot learning or fine-tuning.
    * We trained the model with 10 billion parameters on a 4TB corpus consisting of plain texts and a large-scale knowledge graph.

* Empirical results show that the model outperforms the state-of-the-art models on 54 Chinese NLP tasks,
* and its English version achieves the first place on the SuperGLUE [3] benchmark (July 3,  2021),
* surpassing the human performance by +0.8% (90.6% vs. 89.8%).

* problems
    * However, these large-scale pre-trained language models with hundreds of billions of parameters are trained on plain texts.
        * Such raw texts lack explicit representation of knowledge such as linguistic knowledge and world knowledge.
    * In addition, most large-scale models are trained in an auto-regressive way, but [6] shows that such models demonstrate poorer performance with traditional fine-tuning when adapting to downstream language understanding tasks.

* to solve the problem caused by a single auto-regressive framework and to explore the performance of knowledge enhanced pre-trained models with large-scale parameters, we propose a unified framework called ERNIE 3.0 to train large-scale knowledge enhanced models on a 4TB corpus consisting of plain texts and a large-scale knowledge graph by fusing the auto-regressive network and the auto-encoding network.
* The proposed ERNIE 3.0 can handle both natural language understanding tasks and natural language generation tasks through zero-shot learning, few-shot learning or fine-tuning.
* Furthermore, the proposed framework supports the introduction of various customized tasks at any time.
    * These tasks share the same encoding networks and are trained through multi-task learning.
* when given a new task, our framework could incrementally train the distributed representations based on the previous training parameters, with no need to train them from scratch.

3 ERNIE 3.0

* figure 1 - architecture
* the pre-training tasks spread three task paradigms,
    * natural language understanding,
    * natural language generation 
    * knowledge extraction.
* Therefore, ERNIE 3.0 innovatively designs a Continual Multi-Paradigms Unified Pre-training Framework to enable the collaborative pre-training among multi-task paradigms.

3.1 Overview of ERNIE 3.0 Framework

* Unlike the prevalent unified pre-training strategy of employing a shared Transformer network for different well-designed cloze tasks and utilizing specific self-attention masks to control what context the prediction conditions on,
* ERNIE 3.0 designs a new Continual Multi-Paradigms Unified Pre-training Framework.
    * We believed that 
        * the different task paradigms of natural language processing depend on identical underlying abstract features consistently, such as lexical information and syntactic information, 
        * but the requirements of top-level concrete features are incompatible, in which the natural language understanding tasks have the disposition to learn the semantic coherence while natural language generation tasks expect further contextual information.
    * Therefore, inspired by the classical model architecture of multi-task learning, in which the lower layers are shared across all tasks while the top layers are task-specific,
    * we proposed the ERNIE 3.0 to enable the different task paradigms to share the underlying abstract features learned in a shared network and utilizing the task-specific top-level concrete features learned in their own task-specific network respectively.
    * ERNIE 3.0 exploits the continual multi-task learning framework introduced in ERNIE 2.0 [33].
    * We refer to the backbone shared network and task-specific networks as the Universal Representation Module and Task-specific Representation Modules.
    * mitigates the dilemma that large-scale pre-trained models are difficult to implement with limited time and hardware resources,
        * it permits the models to only update the parameters of a task-specific representation network during the fine-tuning phase.
    * ERNIE 3.0 employs the collaborative architecture of a Universal Representation Module and two Task-specific Representation Modules, namely natural language understanding (NLU) specific representation module and natural language generation (NLG) specific representation module.

3.1.1 Universal Representation Module

* uses a multi-layer Transformer-XL [34] as the backbone network
* We refer to the backbone as Universal Representation Module and it is shared across all the task paradigms.
* applied larger model
* what needs special attention is that the memory module is only valid for natural language generation tasks while controlling the attention mask matrices.

3.1.2 Task-specific Representation Module

* the task-specific representation module is also a multi-layer Transformer-XL,
* which is used to capture the top-level semantic representations for different task paradigms.
* ERNIE 3.0 sets the task-specific representation module to a manageable size, that is a base model size, instead of the multi-layer perceptron or shallow Transformer commonly used in multi-task learning,
* which will produce three obvious benefits,
    * the first is that the base network has a stronger ability to capture semantic information than multi-layer perceptron and shallow Transformer; 
    * the second is that the task-specific networks with base model size enable ERNIE 3.0 to distinguish the top-level semantic information among different task paradigms without significantly increasing the parameters of a large-scale model; 
    * the smaller model size of a task-specific network than a shared network

* ERNIE 3.0 constructs two task-specific representation modules: NLU and NLG
    * in which the former is a bi-directional modeling network while the latter is a uni-directional modeling network.

3.2 Pre-training Tasks

3.2.1 Word-aware Pre-training Tasks

* Knowledge Masked Language Modeling 
    * ERNIE 1.0 [7] proposed an effective strategy to enhance representation through knowledge integration, namely Knowledge Integrated Masked Language Modeling task.
    * It introduced phrase masking and named entity masking that predict the whole masked phrases and named entities to help the model learn the dependency information in both local contexts and global contexts.
* Document Language Modeling 
    * Generative pre-training models usually utilize traditional language model or sequence-to-sequence language model as the pre-training task,
    * ERNIE 3.0 opt for traditional language model as the pre-training task to abate the network complexity and heighten the effectiveness of unified pre-training.
* to enable the NLG network of ERNIE 3.0 to model longer text, we introduce the Enhanced Recurrence Memory Mechanism proposed in ERNIE-Doc [37], 
    * which can model a larger effective context length than traditional recurrence Transformer by changing the shifting-one-layer-downwards recurrence to the same-layer recurrence.

3.2.2 Structure-aware Pre-training Tasks

* Sentence Reordering 
    * which is introduced in ERNIE 2.0 [29],
    * aims to train the model to learn the relationship between sentences by reorganizing permuted segments.
    * At length, a given paragraph is randomly split into 1 to m segments during pre-training and all of the combinations are shuffled by a random permuted order.
    * Then, the pre-trained model is asked to reorganize these permuted segments,
    * modeled as a k-class classification problem where k = sum_n_to_m(n!)
* Sentence Distance 
    * which can be modeled as a 3-class classification problem.
    * The three categories represent that the two sentences are adjacent, nonadjacent but in the same document and from two different documents respectively.

3.2.3 Knowledge-aware Pre-training Tasks

* Universal Knowledge-Text Prediction  (UKTP) task,
    * which is an extension of knowledge masked language modeling.
    * While knowledge masked language modeling only requires unstructured texts, universal knowledge-text prediction task requires both unstructured texts and knowledge graphs.
    * figure 2
    * Given a pair of triple from knowledge graph and the corresponding sentence from encyclopedia,
        * we randomly mask relation in triple or words in a sentence.
    * To predict the relation in the triple,
        * the model needs to detect mentions of head entity and tail entity and determine semantic relationship that holds between them in the corresponding sentence.
    * The essence of this process is similar to the distant supervision algorithm [40] in relation extraction tasks.
    * The distant supervision algorithm assume that if two entities participate in a relation, any sentence that contain those two entities might express that relation.
    * to predict words in the corresponding sentence, the model not only considers the dependency information in the sentence, but also logical relationship in the triple.
    * Specifically, the procedure of obtaining pairs of a triple and this corresponding sentence is as follows:
        * given a document from encyclopedia,
        * we first find the candidate triples in the knowledge graph whose mentions of head entity or tail entity is title of the document,
        * and then select triples from candidate triples whose mentions of head entity and tail entity are mentioned in the same sentence in the document.

* ERNIE 3.0 
    * trains the NLU network through knowledge masked language modeling to improve the capacity of capturing the lexical information,
    * trains the sentence reordering task and the sentence distance discerning task to strengthen the ability of capturing the syntactic information,
    * and finally optimizes the model with the universal knowledge-text prediction task to improve knowledge memorization and reasoning.
    * trains the NLG network with the document language modeling task to enable various generation styles.

3.3 Pre-training Process

3.3.1 Pre-training Algorithm

* Progressive training was originally proposed to improve stability, which starts from an efficient and small model and gradually increase the capacity [41].
* Preliminary application of progressive training has been made on Transformer pre-training.
* BERT([6]) designs a two-stage training with a reduced sequence length for the first 90% of updates.
* [15] also gradually increase the batch size linearly from a small value to the full value.
* we propose to adjust the training regularization factors in a more comprehensive and smooth way  by progressively and simultaneously increasing the training factors including the input sequence length, the batch size, the learning rate and the dropout rate.
* In fact, it is common that Transformer models adopts the learning rate warm-up strategy to increase training stability and our improved progressive learning strategy is compatible to the existing strategy.

3.3.2 Pre-training Data

 * we construct a large-scale,  wide-variety and high-quality Chinese text corpora amounting to 4TB storage size in 11 different categories.
  * To our best knowledge,  this is currently the largest Chinese pre-training corpora compared with 
    * CLUECorpus2020 [45] (100GB),
    * Chinese multi-modal pre-training data [21] (300GB),
    * WuDaoCorpus2.0 used by CPM-2 [20] (2.3TB Chinese data and 300GB English data) 
    * PanGu Corpus [22] (1.1TB).

* we build the corpus for ERNIE 3.0 based on that 
    * from ERNIE 2.0 (including baike, wikipedia, feed and etc),
    * Baidu Search (including Baijiahao, Zhidao, Tieba, Experience),
    * Web text,
    * QA-long,
    * QA-short,
    * Poetry 2&Couplet 3,
    * Domain-specific data from medical, law and financial area 
    * Baidu knowledge graph with more than 50 million facts.

* To improve the data quality, we adopt the following pre-processing strategies:
    * Deduplication 
        * is conducted on different granularities including character level, paragraph level and document level.
        * On the character level, we replace consecutive identical characters (i.e., spaces, tabs, exclamation mark, question mark and etc) with one single character.
        * One the paragraph level, we replace two identical consecutive paragraphs consisting of N sentences with one single paragraph where 0 < N < 100.
        * The two aforementioned deduplication strategies are critical for ERNIE 3.0 to generate non-repeating contents.
        * we adopted Message Digest Algorithm5 (MD5) to filter duplicate documents by comparing the sum of the MD5 of top-3 longest sentences from each document.
    * Sentences with less than 10 words are filtered since they may be problematic or incomplete ones which contains limited semantic information for model pre-training.
    * We further conduct sentence segmentation using regular expressions and word segmentation based on Baidu?s word segmentation tool.

* Then, each dataset is multiplied by a user-defined multiplier number to increase the data diversity after truncating the data for NLU-network pre-training.

3.3.3 Pre-training Settings

* use Transformer-XL[34] structure as the backbone.
* For the universal representation module
    * 48 layers, 
    * 4096 hidden units 
    64 heads. 
* For the task-specific representation modules
    * 12 layers, 
    * 768 hidden units 
    * 12 heads. 
* The total parameter of universal representation module and task-specific representation modules is 10 billion. 
* The activation function used is GeLU[46]. 
* The maximum sequence length of context and the memory length of language generation is set to 512 and 128, respectively. 
* The total batch size of all pre-training tasks is set to 6144. 
* We use Adam[47] 
* learning rate of 1e-4, beta_1 = 0:9, beta_2 = 0:999, 
* L2 weight decay of 0.01, 
* learning rate warmup over the first 10,000 steps 
* linear decay of the learning rate. 
* In the first 10,000 steps, we also use the progressive learning to speedup convergence in the initial stage of pre-training. 
* The model is trained for a total of 375 billion tokens 
* 384 NVDIA v100 GPU cards 
* is implemented on PaddlePaddle framework.
* By virtue of parameter sharding used in [48, 49], we manage to reduce the memory usage of our model and address the problem of the total parameter of model exceeding the memory of a single GPU card.

4 Experiments

4.1 Evaluation Tasks
* We executed extensive experiments on 54 NLP tasks 

4.1.1 Natural Language Understanding Tasks
* 45 datasets belonging to 14 kinds of natural language understanding tasks are used in our experiments, as follows:
    * Sentiment Analysis: NLPCC2014-SC 6, SE-ABSA16_PHNS 7, SE-ABSA16_CAME, BDCI2019 8.
    * Opinion extraction: COTE-BD [50], COTE-DP [50], COTE-MFW [50].
    * Natural Language Inference: XNLI [51], OCNLI [45], CMNLI [45].
    * Winograd Schema Challenge CLUEWSC2020 [45].
    * Relation Extraction: FinRE [52], SanWen [53].
    * Event Extraction: CCKS2020 9.
    * Semantic Similarity: AFQMC [45], LCQMC [54], CSL [45], PAWS-X [55], BQ Corpus [56].
    * Chinese News Classification: TNEWS 10, IFLYTEK [57], THUCNEWS 11, CNSE [58], CNSS [58].
    * Closed-Book Question Answering: NLPCC-DBQA 12, CHIP2019, cMedQA [59], cMedQA2 [60], CKBQA13, WebQA [61].
    * Named Entity Recognition: CLUENER [45], Weibo [62], OntoNotes [63], CCKS2019 14.
    * Machine Reading Comprehension: CMRC 2018 [64], CMRC2019 [65], DRCD [66], DuReader [67], Dureaderrobust [68], Dureaderchecklist, Dureaderyesno15, C3 [69], CHID [70].
    * Legal Documents Analysis: CAIL2018-Task1 [71], CAIL2018-Task2 [71].
    * Cant Understanding: DogWhistle Insider, DogWhistle Outsider[72].
    * Document Retrieval: Sogou-log [73].

4.1.2 Natural Language Generation Tasks

================================================================================
Alquist 4.0: Towards Social Intelligence Using Generative Models and Dialogue Personalization
================================================================================

* The open domain-dialogue system Alquist has a goal to conduct a coherent and engaging conversation that can be considered as one of the benchmarks of social intelligence.
* brings two main innovations: coherence, and engagingness of the conversation.
* coherence
    * we propose a novel hybrid approach combining hand-designed responses and a generative model.
    * The proposed approach utilizes hand-designed dialogues, out-of-domain detection, and a neural response generator.
    * Hand-designed dialogues walk the user through high-quality conversational flows.
    * The out-of-domain detection recognizes that the user diverges from the predefined flow and prevents the system from producing a scripted response that might not make sense for unexpected user input.
    * the neural response generator generates a response based on the context of the dialogue that correctly reacts to the unexpected user input and returns the dialogue to the boundaries of hand-designed dialogues.
* The innovations for engagement that we propose are mostly inspired by the famous exploration-exploitation dilemma.
    * To conduct an engaging conversation with the dialogue partners, one has to learn their preferences and interests?exploration.
    * to engage the partner, we have to utilize the knowledge we have already learned?exploitation.

* We can roughly describe coherence as the ability of the socialbot to correctly understand and advance the dialogue,
*  and engagement as the ability to entertain the other side of the dialogue [4].

* The innovation for coherence is based on the novel combination of hand-designed dialogues and generative models.
    * The hand-designed dialogues are represented as graphs consisting of nodes representing user inputs and bot responses.
    * Nodes are organised into branching flows (Figure 2).
    * The advantage of hand-designed dialogues is that the dialogue designers have a complete control over the flow of the dialogue.
    * This level of control enables them to design high-quality conversations.
    * But on the other hand, the dialogue graphs are rather rigid because they cover only the most common user inputs.
    * Thus, when the user says something that was not anticipated, what we call out-of-domain input, the dialogue reacts with one of the responses that were prepared for the most common inputs.
    * Such a response does not make sense in most cases.
    * The solution to the problem of out-of-domain inputs is made of two steps.
        * The first step is to recognize that the dialogue is not prepared to handle the input.
        * The second step is to produce a new response from the context of the dialogue that is coherent.
        * We apply out-of-domain recognition to the former and the Neural Response Generator to the latter.
            * Out-of-domain (OOD) recognition is a part of the intent classifier that recognizes that a given input is unexpected.
            * The Neural Response Generator is a neural generation model trained on large dialogue corpora that produces a response based on the context of the dialogue.

* The proposed innovation for engagement 
    * is based on the fact that in order to entertain the conversational partner, one has to learn what entertains the partner first and then utilize the knowledge in the following conversation.
    * This might remind us of a famous problem of computer science, the problem of exploration and exploitation.
    * It is safe to say that the socialbot is in the role of an entertainer that has zero prior knowledge about the user. Because the socialbot has zero knowledge,
    * it has to explore the user?s preferences first.
    * Gradually, it has to proceed into an exploitation phase after some time, to maximize its engagement score.
    * For the exploration part, in which Alquist learns the preferences of the user, the main research and development emphasis was put on 
        * Skimmer (section 2; a component that extracts information the user mentions without the bot explicitly asking for it),
        * User Profile (section 3)
        * Entity and Knowledge Utilization (section 4).
        * The mentioned components collect and organize the pieces of information mentioned by the user that are utilized in the following dialogue.
    * For the exploitation part, in which Alquist utilizes the knowledge about the user, the main emphasis was put on 
        * the research and development of the Dialogue Manager (section 5),
        * Trivia Selection (section 6),
        * Intent and Out-of-Domain classification (section 7),
        * the Neural Response Generator (section 8).
        * Those components are responsible for selecting the next action in the dialogue and response production that utilizes the knowledge about the user.

* Figure 1 - We took [22, 20, 21] as our starting point.
    * the Skimmer analyses the user input for the mentioned pieces of information.
    * The pieces of information are stored in the User Profile.
    * Based on the values stored in the user profile, the Dialogue Management selects the next dialogue to start, or selects and presents some trivia related to the actual topic of a conversation.
    * The dialogue is directed according to the Intent classification of the user input.
    * if the Out-of-domain classification recognizes an unexpected user input, the Neural Response Generator produces a coherent response based on the context of the conversation.

2 Skimmer

* One of the critical aspects of each engaging conversation is working with the information mentioned by your communication partner.
* two basic scenarios.
    1. when the bot asks a user a direct question (e.g. ?Do you have a brother??) and stores the answer to the question.
        * In this scenario, the bot is aware of the dialogue context, it knows what type of answer is expected, and it can store the response in the User Profile accordingly.
        * Using the stored value, the bot can carry out a highly personalized conversation and ask relevant questions such as ?How is your brother today??.
    2. Since we do not want to disrupt the fluency of the dialogue by asking too many personal questions to gather information about a user, we want the bot to have the ability to extract the information from each user utterance.
        * We want to extract the information from the sentence regardless of the topic being discussed.
        * For this purpose, we implemented the component called Skimmer.
        * It skims through each utterance and saves the values in User Profile based on a list of rules.
        * Each of the rules contains the following attributes:
            * Regular expression - a set of patterns which must or must not (negative patterns) be contained in the utterance.
            * User Profile attribute - the name of the attribute where the value will be stored.
            * Value - the value stored in the attribute, typically true, false, or a matched group of the regular expression.
* The component processes the user utterance in the following way.
    * It takes each rule from the list and tries to match it with the corresponding regular expression.
    * If it is matched, the value is stored in the specific attribute of the User Profile.

3 User Profile

* User Profile is a unified storage of information which is relevant for the conversation.
* It consists of two main parts?Long-term User Profile and Short-term User Profile, also called Session-scoped Profile.
* The Long-term Profile holds the information about a user across sessions.
* The values stored in the User Profile are used in the Dialogue Selector and additionally in individual dialogues across various topics.

3.1 Profile Structure

* The Long-term User Profile is divided into several sections,
    * mostly corresponding to the topics such as movies,  sports,  books,  etc.
    * The additional sections contain general information about the user not specific to a topic (i.e.,  name,  mood).
 * Each section contains several attributes filled by the corresponding dialogues or by the Skimmer component on the global level.
*  All attributes have a default value representing the state in which the bot does not know the information about the user.
*  When the dialogue reads the default value and needs to work with it further,  it asks the user a question to fill the proper attribute value.
 * The Short-term User Profile stores the entities discussed in the current session.
    * It is mainly used to get the last entity discussed in the conversation or the last entity corresponding to the specific topic.

3.2 Profile Resetting

* From the technical perspective,  each user is identified by the user ID assigned to the specific Echo device.
* However,  multiple users may use the same device and interact with the bot.

* At the beginning of each conversation,  the bot asks for the user?s name,  and the following situations may happen:
    * The user ID has a first session with the bot ? Bot asks for the user?s name and creates a new Long-term Profile.
    * The user ID has had a session before, but the user?s name is not saved? Bot asks whether they have talked to each other before.
        * If the user says ?yes?, the old profile is restored.
        * Otherwise, a new one is created.
        * In both cases, the bot asks for the user?s name additionally.
    * The user ID has had a session before, and the user?s name has a value stored ? Bot asks the user to confirm his name.
        * If it is the same user as the last session, the previous profile is used.
        * Otherwise, a new one is created.

* The Short-term User profile is reset at the beginning of each session.

4 Entity and Knowledge Utilization

* The experience from the previous years of the Alexa Prize competition has shown that to have a meaningful conversation, the socialbot has to be able to chat about specific entities that naturally emerge from the conversation [21].
* the conversation should both include factual information about the entity and be interested and engaged in the user?s feelings and personal experiences about the topic [19].
* we developed a system that builds on top of classical entity detection by linking them to domain-specific publicly accessible databases rather than one general-purpose knowledge base.
* Then we utilize the information from the mentioned databases to continue the conversation.

4.1 Entity Recognition

* We approach entity recognition as a sequence tagging task as described in [20].
* We utilize the BI-LSTM [8] model.
* We train the model on a hand-annotated dataset from the data gathered from the user utterances in the previous years of the competition.
* Each token in the utterance is classified into one of the defined classes, 
    * B-type as the beginning of an entity of a given entity type,
    * I-type as an inside token of a given entity type,
    * O as an outside token.
* We predict 16 entity types,
    * Movie, Sport, Job, Language, Music genre, et cetera.
* The predicted entity and type are then used by the dialogue selector (section 5.    1) to manage the conversation flow.
* after entity recognition, the selected entity types are linked to specialized external knowledge sources.

4.2 Entity Linking and Knowledge Base

* We utilize external public domain-specific databases to obtain additional information about the recognised entity.
    * TMDb2 database - Movie and Person (in the movies topical context) types.
    * LastFM3 database - Song and Person (in the music topical context) types.
    * Goodreads4 database - Book and Person (in the books topical context) types.
    * IGDB5 database - Videogame entity type.
* For these entity types, we query the corresponding database with the recognized entity and from the returned candidates, we then select the one with the highest relevance and popularity based on the database?s search algorithm.
* Each linked entity is then stored in the short-term session context in the User Profile together with the reference to the corresponding knowledge database and the information received from the database with the initial query.
* In certain specific contexts, the entity is also stored in the long-term User Profile context that persists between sessions.
    * This can be, for example, the user?s favorite movie or the user?s last read book.
* whenever there is a need to access the entity in the database again, using a different query to obtain more information is simplified by retaining the entity reference.
* In general, we have found this approach more manageable and with significantly less overhead than creating and maintaining our own dedicated general-purpose knowledge base would lead to.

5 Dialogue Management

* follows the basic principles outlined in the previous iterations of the system [20, 21].
* The conversation consists of several small scripted dialogues, where turn-by-turn interactions in the context of the flow are handled solely by the intent detection component of the system (described in detail in Section 7).
* The novel part of dialogue management within our system concerns primarily the selection of these dialogues.
* We have created a new component called Dialogue Selector, whose purpose is to select the most relevant dialogue following after the previous dialogue has concluded so that the context and the coherence of the conversation remain intact.
* The system takes into account all previously mentioned information that is retained in the User Profile, allowing it to exhibit socially intelligent behavior.

* 5.1 Dialogue Selector

* For the dialogue selector to judge the relevance of the dialogues, each dialogue is described by three types of information.
    * A set of tags that represent what topics the dialogue touches on.
            * For example, the dialogue Esport is tagged as Games and Sport.
    * A set of User Profile attributes that are relevant to the dialogue.
        * For example, the dialogue Favorite Movie has relevant attributes favoriteMovie, discussedMovie, likesMovies.
    * A necessary starting condition that determines whether it is possible to initiate the dialogue within the current context successfully.
        * For example, the dialogue Favorite-Movie-Part requires that there is a movie being discussed with the user,
        * thus the attribute discussedMovie cannot be empty.
        * a common necessary starting condition is that the dialogue has not yet been initiated during the current session.

* The dialogue selector utilizes this information together with the current state of the user profile to select a relevant continuation in the following steps:
    1. If there is relevant trivia for the currently discussed entity/topic and no trivia has been mentioned for three or more dialogues; the system selects the corresponding trivia as the next dialogue.
        * (Experiments in the previous years of Alexa Prize have shown including relevant trivia in the conversation improves the results of the bot [20]).
    2. From all dialogues included in the socialbot, the system filters out all dialogues whose necessary starting condition has not been fulfilled.
    3. If there are no remaining dialogues, the system selects the Neural Response Generator for the continuation of the conversation.
    4. If there are remaining dialogues, the system looks at the dialogue tags that have just finished and compares them to the tags of the available dialogues. It then discards all dialogues except the ones with the highest overlap.
    5. From the new pool of dialogues, the system looks at the relevant attributes of the dialogue that have just finished and compares them to the attributes of the available dialogues. It again discards all dialogues except the ones with the highest overlap.
    6. If the system finds a non-zero overlap either in the dialogue tags or dialogue attributes, it selects the new dialogue randomly from the final dialogue pool.
    7. If the overlap in both dialogue tags and attributes is zero, the system instead tries to find an overlap between the User Profile attributes relevant to this session and the attributes of the available dialogues.
    8. If the system finds a non-zero overlap between the relevant User Profile attributes and the dialogue attributes, it selects the new dialogue randomly from the final dialogue pool. This usually means selecting a new conversation topic in which the user has previously shown interest.
    9. Finally, if no overlap is found, the system initiates a recommendation dialogue designed to help the user select a new topic of the conversation.

6 Trivia Selection

* The experiments done in Alquist 2.0 [20] showed that trivia information (also called fun-facts) about the topics and entities is an essential part of the conversation.
* In Alquist 2.0, the trivia selection procedure was implemented roughly as follows:
    * The trivia facts were scraped from Reddit on a daily basis.
    * During the runtime of the conversation, the trivia database was queried using only the value and type of the currently discussed entities.
    * The most recent trivia was returned.
* This approach suffers from the selection of irrelevant trivia information in specific situations.
    * We try to address this issue using a model estimating the similarity between the trivia text and the recent context of the conversation.

6.1 Model

* The task of the model is formulated as an estimation of similarity between the text of the trivia and n most recent utterance?response pairs.
    * We empirically estimated the value of n to 2.
 
* The full procedure can be described in the following steps.
    1. During the scraping phase, the trivia text is encoded, and the dense representation is stored.
    2. During runtime, a candidate trivia list is retrieved using a full-text search with the entity text as query.
    3. Context of n most recent utterance?response pairs is encoded.
    4. Cosine similarity between the context representation and each candidate is computed, and the most similar one is selected.

* The model architectures we experimented with are shown in Table 1.

6.2 Datasets

* We manually annotated the turns from the conversations gathered during the previous versions of our bot and used the data for fine-tuning.
* Only the turns (and their respective contexts) which contained trivia were considered.
* We created a dataset with 350 turns where the trivia was mentioned.
* Each sample consists of one suitable piece of trivia and four negative examples (randomly selected trivia), a context (list of user utterance?bot response pairs), and an annotation of whether the trivia is relevant given the context or not.
* Three hundred samples were used for actual fine-tuning, and fifty samples were used for testing.

6.3 Results and Discussion

* Table 1
* We experimented with both not fine-tuned and fine-tuned versions.
* Each model scored five candidate pieces of trivia.
* We also compared the results with the DialogRPT ranker used in the Neural Response Generator.
* The results are compared with a baseline approach?the final trivia is selected randomly.

7 Intent and Out-of-Domain Classification

* Following the concept of dialogue presented in Alquist 2.0 [20], we design each dialogue as a tree structure.
* The tool described in [20, 21] is used for designing the dialogues.
* A crutial point in the conversation structure is where we expect the user input/user utterance.
* Each user utterance is then classified into a specific intent for which the dialogue designer manually writes training utterances.
* However, because of the complexity of language and the open-world assumption [11], the dialogue designer cannot incorporate each possible intent.
    * Based on that, these user utterances for which the dialogue is not prepared are called out-of-domain.
* The ID (in domain) intent is a user utterance for which the dialogue designers have prepared a response.
    * Such a response is designed in a coherent and engaging conversational style.

* We have also incorporated the concept of hierarchy into our dialogue design and introduced two types of ID  intents:
    * intents valid across all dialogues (global ID intent) 
    * intents valid only in the specific context (local ID intent).

7.1 Model

* our intents create a hierarchical structure, which can be seen in Figure 2.
* The hierarchical structure provides the dialogue designer with modularity in creating the flow of the conversation but puts a significant emphasis on the effectiveness of the algorithms for intent classification.
* To allow the suggested modularity,  we train a separate model for each level of the hierarchy.
* figure 3
* The system works in two steps.
    1. we need to determine which intent model in the hierarchy is appropriate ? local or global.
        * We utilize cosine similarity over sentence embeddings between user query and train examples of each intent to classify utterances as local or global intents while prioritizing local intents.
        * The priority is based on using a stable sorting algorithm over cosine similarities between sentence embedding of the user utterance and the examples of each intent.
        * We also allow manual setting of the threshold for local and mainly for global intents leading to the filtering out of the intents if the cosine similarity is not high enough.
    2. the utterance is classified into a specific intent by the corresponding logistic regression selected in the previous step.
        * We use logistic regression because of the speed of its training and the proven performance in low-resource scenarios (see Figure 4).

* the cosine similarity can filter out all ID classes (if the similarity score falls below the threshold).
    * It will lead to the output of the OOD class.
    * This is an approach similar to [13].
* to make our system more robust,
    * we include a dynamic threshold similar to [26].
    * Our dynamic threshold is based on the arithmetic mean of similarities between the two closest train examples in each intent.
* This dual-threshold approach balances trade-offs between the manual control of the OOD sensibility and the robustness of the whole system.

7.2 Datasets

* We performed our analysis on a publicly available dataset as well as on manually labelled anonymized queries:
    * CLINC150 Dataset [13] 
    * ALQUIST 4.0 Dataset 

* A summary of the datasets is shown in Table 2 and Table 3.

* All samples in the ALQUIST 4.0 Dataset were carefully checked for consistency and drawn from aggregated and anonymized queries.
    * The datasets generated during this study are not made available for privacy reasons.
* The CLINC150 Dataset was augmented to support the hierarchical structure ? 
    * the augmentation was performed as a random selection of 15 intents from the CLINC150 Dataset 
    * then divided into two sets ? one set represents local intents, and the other represents global intents.
    * The ratio was set to 1:3 to represent a typical situation in a dialogue.
    * The split results in 4 local intents and 11 global intents ? a detailed overview is shown in Table 3.
    * The process was repeated 15 times.
    * The results shown in the experimental section are average.

7.3 Experiments

* Our analysis includes an inspection of the input features (embeddings).
    * Average of word embeddings FastText [18] 
    * Universal Sentence Encoder - Deep Average Network (USE-DAN) [2] 
    * Universal Sentence Encoder - Transformed-encoder (USE-TRAN) [2] 
    
* The model described in subsection 7.1 is tested in two ways ? automatically and manually.
    * The automatic evaluation is performed over our ALQUIST 4.0 Dataset (shown in Table 2) and the artificially hierarchical augmentation of the CLINC150 Dataset (shown in Table 3).
    * The manual evaluation was performed on aggregated data selected from anonymized user conversations with the socialbot.

* We collected all user utterances from these parts of dialogues and performed the human evaluation.
* The results can be seen in Table 4.
* Besides evaluating the performance solely for the OOD detection, we looked at the performance of the local and global intents.
* To select the most suitable model for the final intent classification, we measure the difference between the three most common classification models?Logistic Regression, Support Vector Machine, and the 2-layer Neural Network.
* We focus on the necessary number of needed examples to achieve sufficient accuracy.
* The evaluation is shown in Figure 4 and highlights the problem of the neural network when dealing with low-resource scenarios.
* The measurement was performed over CLINC150, randomly choosing five classes (our average number of intent classes for the intent model) and randomly choosing N examples.
 * *This procedure was repeated 25 times,and then shown values are average.
W* e selected Logistic Regression as a model performing well in the low-resource scenario.

7.4 Results and Discussion

* our results (shown in Table 5) suggest a relationship between the type of embedding and the performance of the classification model.
* The difference in results can be explained by the word-order sensitivity of advanced embedding techniques.
* Another important aspect is the memory requirements and the speed of obtaining the embedding (shown in Table 6).
* The results suggest the usage of USE-DAN as an appropriate embedding layer.
* It should be mentioned that the performance on each level of the hierarchy remarkably differs between our two datasets.
* We believe that it is caused by the artificial augmentation of the CLINC150 Dataset and its unrealistic representation of the real-world use cases.
* We should also notice the higher performance for the local intent classification than for the global intent classification (notable mainly on the ALQUIST 4.0 Dataset).
    * It is caused by the hierarchical structure of our model, which emphasizes the local intent over the global as was described in 7.1.
    * This is aligned with our experience that staying in the local context of the dialogue is beneficial for coherence.
* The performance of the OOD detection needs to be evaluated with respect to precision and recall.
    * The high precision and lower recall indicate that the algorithm is suitable for classifying OOD in the conversational domain because we prefer false negative over false positive - it supports the consistency of the dialogue.
* we performed a human evaluation (shown in Table 4) which demonstrates the performance on real-world data.
    * It supports previously stated conclusions.

8 Neural Response Generator

* It generates a response based on the most recent turns of dialogue.
* We use such a model in Alquist in two settings.
    * The neural response generator creates a response for out-of-domain user inputs,
    * it generates follow-up questions about trivia.
* The motivation to use a Neural Response Generator for the out-of-domain (OOD) inputs is the following.
    * We put the main content emphasis in Alquist 4.0 on hand-designed dialogues.
    * Dialogues are represented as graphs (Figure 2).
    * the designer can not predict all possible flows the conversation can go through.
    * We can detect the situations in which the user diverts from the predesigned flow by the out-of-domain detection.
    * However, the problem of how to continue in the conversation in a meaningful way emerges.
        * Neural response generators that can create a response on the fly based on the context of the dialogue and the user input are the way we solve the problem.

* To enhance the conversation with interesting and surprising pieces of information, we use crawled trivia from Reddit6.
    * The trivia has a form of a statement.
    * This property makes incorporating them into the dialogue in a conversational way challenging because statements don?t encourage the user to continue in the conversation as questions do.
    * The solution is to concatenate a follow-up question to the trivia.
    * Because there is a large number of trivia, it is intractable to write a follow-up question for each piece.
    * Thus, we use a Neural Response Generator to create follow-up questions.

* The practical application of a Neural Response Generator faces several challenges.
    * the generated responses have to be quality enough.
    * the response must be generated quickly enough to be applicable to a conversation in real-time and with reasonable computational resources.
    * the lack of control over the generated response is a factor that limits the application of the Neural Response Generator in combination with hand-designed dialogue content.
    * One of the most significant obstacles we identified is randomness, in which the generator produces questions and statements.
        * The main problem is that questions encourage the user to respond, whereas statements do not.

8.1 Model

* We selected DialoGPT [30] for our experiments.
* It is a large, tunable neural conversational response generation model trained on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 to 2017.
* DialoGPT is based on GPT-2 [23].
* There are three sizes of the model:
    * small with 117M parameters,
    * medium with 345M parameters,
    * and large with 762M parameters.

* DialoGPT optimizes the conditional probability
    * check formula: P(T|S) - seems a language model formula
    * where we concatenate all dialogue turns within a dialogue context into a long text x_1, ..., x_N
    * N is a sequence length). 
    * Each dialogue turn is followed by a special end-of-speech token. 
    * We denote the dialogue history as S = x_1, ..., x_m 
    * and the target sentence as T = x_m+1, ..., x_N

* To have control over whether the model generates a question or a statement, we modified the original DialoGPT.
    * We introduced special tokens to the beginning of the input to DialoGPT.
    * Special tokens QUESTION or STATEMENT are prepended to the dialogue context.
    * They give information to the model, whether our desired response should be a statement or a question.
    * we optimize the conditional probability
        * check formula - seems same as above but includes a command: questions or statement

* DialoGPT can produce several candidate responses. And because some replies are more engaging than others, spawning more follow-up interactions, we have to select the optimal one.
* We employed DialoRPT [6] for this task.
    * DialoRPT is a set of GPT-2 based ranking models trained on 133M pairs of human social media feedback data (number of replies and upvotes) built for feedback prediction.
    * There are five types of rankers:
        * updown that predicts how likely the response gets the most upvotes,
        * width that predicts how likely the response gets the most direct replies,
        * depth that predicts how likely the response gets the longest follow-up thread,
        * human_vs_rand that predicts how relevant the response is for the given context,
        * and human_vs_machine that predicts how likely the response is human-written rather than machine-generated.
    * DialoRPT takes the generated candidate responses and produces a score for each.
        * We select the response that has the largest score.
        * This way, we get the best response according to the selected DialoRPT model.

8.2 Datasets

* Alquist Dialogue Graphs 
    * consist of the dialogue graphs introduced in [20] and represents the human-designed dialogue flows of the socialbot.
    * It consists of 640k dialogues generated out of 80 dialogue graphs.
    * those dialogues have a relatively small semantic diversity.
* Topical-Chat [7] 
    * is a knowledge-grounded human-human conversation dataset where the underlying knowledge spans eight broad topics, and conversation partners don?t have explicitly defined roles.
        * It consists of 10k conversations and 235k utterances.
* EmpatheticDialogues [24] 
    * is a dataset of 25k dialogues grounded in situations prompted by specific emotion labels.
* DailyDialog [15] 
    * is a multi-turn dialogue dataset that reflects daily communications and covers various topics about everyday life.
    * The dataset is manually labelled with communication intention and emotion information.
    * It contains 13k dialogues.
* Merged datasets Dataset 
    * consisting of all dialogues taken from Alquist Dialogue Graphs, Topical-Chat, DailyDialog, and Persona-Chat [29].

* Some of the used datasets contain additional annotations, like emotions, knowledge, situation description, or information about the speaker.
* However, we decided to not use the annotations in our experiments because those annotations are not available in our system in real traffic.
* On the other hand, we modified the dataset to include special QUESTION and STATEMENT tokens.
    * First, we split the turns into sentences by the NLTK [1] sentence tokenizer.
    * Second, Standford CoreNLP [16] annotated each sentence as a statement or a question.
    * Third, if the turn contained both statements and questions, we divided the turn into several turns where each turn contains sentences of the same type, and we did not modify the order of sentences.
    * Lastly, we label all turns consisting of statements by a special token STATEMENT and all turns consisting of questions by a special token QUESTION.

8.3 Evaluations

* We evaluated 
    * small, medium, and large DialoGPT models 
    * with the STATEMENT and QUESTION token modification
    * trained on five datasets.

* The models used beam search with five beams.
* Each of the models generated five alternative responses that were ranked by the updown DialoRPT model and we took the top response for evaluation.
* We evaluated the models in three metrics.
    * Question/Statement accuracy evaluated models ability to generated questions and statements correctly based on the specification of the former or the latter.
        *We fixed 1000 three turn long contexts as inputs.
        * For each of them, we generate five responses with the QUESTION token and five with the STATEMENT token.
        * Next, Stanford CoreNLP annotated all generated responses as a question or a statement.
        * In total, we have 10,000 examples for which we compute the accuracy.
        * Next, we performed the human evaluation of the model outputs.
    * For OOD, we took 100 dialogue contexts (3-5 utterances long) and generated the following response in the form of a question.
        * The responses were manually labelled as OK or not.
        * Annotation OK labels the response that correctly progresses in the conversation flow.
        * Responses that we did not label as OK were unrelated to the given context,
    * For trivia,
        * we took 100 pieces of trivia crawled from Reddit.
        * We used the trivia as an input to the DialoGPT model, and we generated the follow-up question.
        * We also labelled responses as OK or not.
        * We further divided Not OK responses into unrelated to the given trivia or factually untrue.
        * We did not use the repeated class as the only input to the model was the trivia.
        * Thus, the model could not know that the given question was asked in the previous turns of the conversation.
    * We evaluated the inference time of the models as well,
        * We evaluated the models on CPU as well as GPU for several numbers of generated responses.
        * The CPU was Intel(R) Xeon(R) Platinum 8175M CPU @ 2.50GHz used in AWS EC2 instance m5.2xlarge.
        * The GPU was Tesla T4 used in AWS EC2 instance g4dn.4xlarge.
        * We fixed the input to the model and averaged the time of 20 inferences.

8.4 Results

* Table 8 presents the comparison of the small, medium, and large DialoGPT models trained on five dialogue datasets [9].
    * We can notice that all models on all datasets possess a good ability to produce questions or statements based on the desired output.
    * The only dataset on which models do not get over a 90% threshold is EmphateticDialogues.
    * Next, we can notice that the larger the model is, the better responses it produces based on human evaluation on both OOD and trivia.
    * The large DialoGPT model trained on EmpatheticDialogues produces the best responses for out-of-domain.
    * The large DialoGPT model trained on Topical-Chat produces the best follow-up questions for trivia.

* We also present the results of the error analysis in Table 9 [9],
    * where we assigned each generated response to one of the three error classes (unrelated, untrue, and repeated) we identified in the case of OOD 
    * and two error classes (unrelated, and repeated) in the case of funfacts.
    * We can see that the two biggest issues for all models in the case of OOD are unrelated and repeated responses.
    * The situation is similar for trivia.
    * The bigger problem than untrue are unrelated responses.
    * To the best of our knowledge, there is no easy way to solve unrelated responses.
    * The repeated responses might be filtered using semantic text similarity, but further research is needed.
    * The untrue responses are mainly a problem of the Topical-Chat dataset because of its nature.
        * It is a dataset where two dialogue agents have conversations about trivia.
        * Thus, we hypothesise that the model learns to generate responses that resemble trivia,
        * but the model is not powerful enough to learn true trivia only.
        * However, we identified that the model tends to start such responses by a phrase Did you know.
        * Thus we can use string matching to remove such responses in practice.

* Table 10 presents the results of the experiment evaluating the inference times of small, medium, and large models [9].
    * The two obvious facts we can notice are that the larger the model is and the more variants it generates, the longer the inference time is.
    * We can also see that the inference time is significantly shorter on GPU than on CPU.
    * accuracy vs time problem
    * We selected 400 ms as a time threshold we do not want to cross as more time poses a noticeable time delay in the conversation when we sum the processing time of the rest of the system?s components.
    * Next, we decided that we want as good responses as possible.
    * Because the GPU cost was manageable for us, we selected the largest model running on GPU, generating five variants of response.
    * Later in the competition, we switched to three variants because the processing time of the rest of the system increased due to larger complexity.

8.5 Examples

* Table 11 presents generated questions for OOD.
    * We kept only the last OOD input from the three-turn context that the model receives as input for clarity in Table 11.
    * However, part of the generated questions are statements.
    * This fact was hinted in the Question/Statement accuracy on the EmpatheticDialogues dataset in Table 8.
    * Although it is an error, we do not consider it as critical because the generated response continues the dialogue appropriately.

* Table 12 presents the generated questions about trivia.
    * We can notice that in the first example the model understands that the words book and read are related,and for the book-related trivia generates a question Have you read it? that correctly addresses the user too.
    * We can observe a similar phenomenon in the next three examples, where the model additionally understands that Brazil, Bill Murray, and Caesar salad are the primary entities of trivia.
    * Finally, in the second to last example, we can see that the model also addresses itself by a philosophically-looking question.
        * The generated follow-up questions of this model look reasonable,and we hypothesise that it is because of the constrained domain of trivia.

8.6 NRG Conclusion

* We applied a Neural Response Generator to Alquist to generate responses for user inputs that are outof- domain of our hand-designed dialogues and to generate follow-up questions to trivia to encourage a user to continue in the dialogue.
* We selected the DialoGPT model that generates several response variants, followed by the DialoRPT model that ranks them.
* A lack of control over the model?s output, mainly whether the model generates a statement or a question, made it challenging to incorporate the model into handmade dialogues.
    * For this reason, we modified DialoGPT to condition the response on special tokens STATEMENT and QUESTION.
* The quality of responses, ability to control the output, and inference time were crucial in our system.
* We used Alquist Dialogue Graphs, DialyDialog, Topical-Chat, EmpatheticDialogues, and Merged datasets for evaluations.
    * We modified the datasets by dividing their turns and labelling them by the STATEMENT and QUESTION labels.
* We evaluated small, medium, and large DialoGPT models.
* The experiments showed that
    * the best performance on the OOD task achieved a large DialoGPT model trained on the EmpatheticDialogues dataset 
    * and the best performance on trivia achieved a large DialoGPT model trained on Topical-Chat dataset.
    * the question/statement accuracy was above 90% for most models which we consider a good result.
    * From the point of view of inference time, the best option still tractable is the large DialoGPT model that generates five variants of responses below 400 ms using GPU.
* The examples of generated responses show that the model can generate relevant responses that contain the entities mentioned in the input to the model.
    * In the case of follow-up trivia questions, it understands the entity (book) and the related activity (to read), can utilize the primary entity of trivia and correctly asks for user?s preferences.

9 Conclusion

* In order to actively engage the user in the conversation, we have developed Skimmer, a component that learns about the user from their messages and fills in information in their User Profile.
    * We are then able to utilize what we have learned about the user?s interests and personality further in the conversation, making it evident that the socialbot is invested in learning more about the user, remembers their preferences, and takes them into account during the conversation.
* We have introduced the out-of-domain query detection as a core functionality of our system.
    * This allows us to hand over more control of the conversation to the user, which makes the socialbot seem more responsive to the user queries.
    * However, it also introduces a problem of how to handle the OOD responses.
* In order to keep the conversation engaging even when moving in the unexpected direction, we have also integrated two generative models into the system based on DialoGPT.
    * Firstly, the Neural Response Generator triggers when the user input is classified as OOD.
    * Secondly, we utilize the NRG to generate a follow-up prompt when presenting the user with trivia relevant to the currently discussed topics.
    * We have shown that in both cases the NRG is able to generate a relevant continuation of the conversation.
    * This allows the system to efficiently handle unexpected responses and previously unseen information and integrate them into the conversation while maintaining the flow of the conversation and its coherence.
