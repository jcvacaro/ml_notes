================================================================================
Recipes for building an open-domain chatbot
https://arxiv.org/abs/2004.13637
================================================================================

* we provide recipes for building opendomain chatbots that perform well in human evaluations.
* Beyond simply scaling models the two main takeaways from our study are: 
    1. Blending Skills
        * Large improvements can be made by finetuning on data that emphasizes desirable conversational skills. 
        * achieving large gains by using the recently introduced Blended Skill Talk (BST) set-up (Smith et al., 2020), 
        * which targets those aspects by providing training data and initial conversational context (personas and topics). 
        * Small models using BST can match or outperform larger models that do not. 
        * While BST emphasizes desirable traits, we also show this tuning can minimize undesirable traits learnt from large corpora, such as toxicity.
    2. Generation Strategies
        * The choice of decoding algorithm is of critical importance, and two models with the same perplexity but different decoding algorithms can give vastly different results. 
        * we show that the length of the bot?s utterances are crucial to human judgments of quality ? too short and the responses are seen as dull or showing a lack of interest, too long and the bot appears to waffle and not listen. 
        * Previous work reported that beam search is inferior to sampling (Holtzman et al., 2019; Adiwardana et al., 2020), 
        * Contrary to that , we show that careful choice of search hyperparameters can give strong results by controlling trade-offs (constraining the minimum beam length ).


* In human evaluations of engagingness our best model outperforms Meena (Adiwardana et al., 2020) in a pairwise comparison 75% to 25%, and in terms of humanness by 65% to 35% (both statistically significant, two-tailed binomial test, p < 0:01).
* However, there are problems:
    * our models still display: a lack of in-depth knowledge if sufficiently interrogated; 
    * a tendency to stick to simpler language; 
    * a tendency to repeat oftused phrases. 

* Model architectures (2)
    * We consider three types of architectures in this work: retrieval, generative, and retrieve-and-refine models. 
    * All three use Transformers (Vaswani et al., 2017)
    * Retriever (2.1)
        * Given a dialogue history (context) as input
        * retrieval systems select the next dialogue utterance by scoring a large set of candidate responses and outputting the highest scoring one. 
        * Typically, all possible training set responses are used as the candidate set.
        * We employ the poly-encoder architecture of (Humeau et al., 2019). 
        * Poly-encoders encode global features of the context using multiple representations (n codes, where n is a hyperparameter), which are attended to by each possible candidate response, see Figure 2. 
        * We consider two poly-encoder sizes: 256M (from (Smith et al., 2020)) and 622M parameter models which we trained here, both using N = 64 codes.
    * Generator (2.2)
        * We employ a standard Seq2Seq Transformer architecture to generate responses rather than retrieve them from a fixed set. 
        * Our implementation is based on the ParlAI version (Miller et al., 2017). 
        * We use Byte-Level BPE tokenization (Radford et al., 2019) trained on the pre-training data, as implemented in HuggingFace?s Tokenizers
        * We consider three sizes of model: 90M parameters (following Shuster et al., 2019), 2.7B parameters and 9.4B parameters. 
        * Our 9.4B parameter model has a 4 layer encoder, a 32 layer decoder with 4096 dimensional embeddings, and 32 attention heads. 
        * Our 2.7B parameter model roughly mimics the architectural choices of Adiwardana et al. (2020), with 2 encoder layers, 24 decoder layers, 2560 dimensional embeddings, and 32 attention heads.
    * Retrieve and Refine (2.3)
        * Current generative models are known to have issues with producing dull and repetitive responses which are improved, but not resolved, by simply scaling (Holtzman et al., 2019; Welleck et al., 2020; Li et al., 2019a). 
        * generative models are known to hallucinate knowledge, and in general are unable to read and access external knowledge other than what is embedded in their model parameters, which may be imperfect. 
        * One approach to try to alleviate these problems is to combine a retrieval step before generation, referred to as a retrieve and refine model (Weston et al., 2018). 
        * We consider two variants for the retrieval step: dialogue retrieval and knowledge retrieval.
        * Dialogue Retrieval 
            * We can simply use a retrieval-based dialogue model in the retrieval step, as in Sec. 2.1. 
            * Given the dialogue history, the retrieval model is first used to produce a response. 
            * Rather than showing this response to the speaking partner it is appended to the input sequence of the generator, along with a special separator token.
            * The generator then outputs a response as normal given this modified input sequence. 
            * Retrieval models produce human written utterances which tend to include more vibrant language than the most high probability utterances of a standard generative model. 
            * Hence, if the generative model learns when to copy the elements of such an utterance, and when not to, it can provide improved responses. 
            * To build such models, we use the architectures considered in the previous two sections for the two components of the model.
        * Knowledge Retrieval 
            * We can also use the same mechanism to first retrieve from a large knowledge base, instead of retrieving an initial dialogue utterance.
            * We can then condition the generation on the retrieved knowledge, as done in models proposed for the Wizard of Wikipedia task (Dinan et al., 2019c). 
            * We use the same retrieval system as in that cited work, which uses a TF-IDF-based inverted index lookup over a Wikipedia dump to produce an initial set of knowledge candidates. 
            * A Transformer retriever model (the same as Sec. 2.1) is then used to rank the candidates and select a single sentence which is used to condition generation. 
            * We additionally trained a Transformer-based classifier to choose when to perform retrieval or not on a per-turn basis, as some contexts do not require knowledge. 

* Training Objectives (3)
    * Ranking for Retrieval (3.1)
        * To train the retrieval models, a cross-entropy loss is minimized in which the logits are ycand_1,...,ycand_n
        * where ycand_1 is the score of the correct response and the others are sampled negatives. 
        * Following Humeau et al. (2019), during training we use the other responses in the batch for negatives.
            * This allows for much faster training, as we can reuse the embeddings computed for each candidate, and also use a larger batch size. 
        * use batches of 512 elements.
    * Likelihood Training for Generation (3.2)
        * To train the generative models, we use the standard Maximum Likelihood Estimation (MLE).
        * check formula - language model equation to predict next workd based on previous context
    * blending for Retrieve and Refine (3.3)
        * simply appending dialogue retrieval responses to the context of a generative model and training with MLE unfortunately does not yield satisfying results. 
        * As the correspondence between gold label and retrieved utterance is not necessarily clear, a trained model often opts to simply ignore the retrieval utterance, as was shown in Weston et al. (2018).
        * To ensure it is used, one can replace the retrieved response instead with the gold response alpha_% of the time, treating alpha as a hyperparameter
        * This gives a smooth transition between retrieval and generator-only systems. 
        * For knowledge retrieval we find this issue to be less of a problem as the fine-tuning datasets used have a clear correspondence between gold knowledge conditioning and response, and in that case we only use the gold knowledge during training.
    * Unlikelihood training for generation (3.4)
        * An alternative method to combat the failures in model generations is to change the loss function.
        * The unlikelihood loss (Welleck et al., 2020; Li et al., 2019a) has been shown to help fix mismatches between human and model distributions across various axes, including decreasing repetitions and mitigating the issue of overrepresented vocabulary tokens.
        * It penalizes a set of tokens C_t at each time-step
        * check equation - log(1 - language model probability)
        *  the overall objective in unlikelihood training then consists of mixing the likelihood and unlikelihood losses
        * check formula - L_MLE + (alpha * L_UL)
        * Likelihood tries to model the overall sequence probability distribution, while unlikelihood corrects for known biases. 
        * Likelihood pushes up the probability of a gold token y, while unlikelihood pushes down the probability of negative candidate tokens C_t
        * It does this via the set of negative candidates C_t calculated at each step t;
        * typically one specifies in advance a method for generating such candidates, for example the tokens which have been repeated or overrepresented. 
        * In this work during training we keep a running count of the distribution of n-grams that appear when generating from the model, and choose tokens as negative candidates from these n-grams when their counts are above the human distribution counts as measured from the gold responses.

* Decoding (4)
    * Beam Search (4.1)
        * Two widely used deterministic decoding approaches are greedy search and beam search. 
        * The former can be seen as a special case of the latter.
        * Greedy search selects the highest probability token at each time step
        * Beam search maintains a fixed-size set of partiallydecoded sequences, called hypotheses. 
        * At each time step, beam search forms new hypotheses by appending each token in the vocabulary to each existing hypothesis, scoring the resulting sequences then selecting the highest scoring sequences.
    * Sampling (4.2)
        * An alternative is to sample from a model-dependent distribution at each step, 
        * In order to prevent sampling low probability tokens, a typical approach is to restrict sampling to a subset of the vocabulary at each step, and sampling according to those (renormalized) probabilities.
        * two methods
        * top-k sampling (Fan et al., 2018) 
        * sample-and-rank (Adiwardana et al., 2020). 
            * performs sampling S times, and selects the generated sample with the highest probability.
    * Response Length (4.3)
        * Generating with a beam tends to produce short generations that do not match the length statistics of the human utterances they were trained on (Weston et al., 2018). 
        * longer responses may expose the model failures
        * two methods
        * Minimum length:
            * the end token is forced to not be generated until a minimum sequence length is achieved.
        * Predictive length 
            * predict the length based on human-human conversation data. 
            * train a 4-class classifier by binning the lengths of the next conversation turn (e.g., < 10, < 20, < 30, or > 30 tokens). 
            * We use the same architecture as the retrieval model for this classifier. 
            * Then, at test time, the classifier is
                * first used to predict the length of the next response,
                * sets the minimum generation length constraint to its corresponding prediction. 
            * good, but more complex
    * Subsequence Blocking (4.4)
        * Sequence generation models are known to repeat subsequences (Holtzman et al., 2018)
            * particularly in stochastic methods such as beam search, but also in sampling methods as well (Adiwardana et al., 2020). 
        * We implement standard beam blocking of n-grams (Paulus et al., 2017) and use n = 3. 
        * We consider both blocking repeated n-grams within the generated utterance, and repeating of the input sequence (previous utterances from either speaker).

* Training Details (5)
    * Pre-training Ranking models. 
        * We perform pretraining using the Fairseq (Ott et al., 2019) toolkit.
        * Our 256M parameter ranking model is identical to the pre-trained model released by Humeau et al. (2019). 
        * Our 622M model is pre-trained using a simple Masked Language Model objective on the same data and dictionary as the large Generative models. 
            * We took all hyperparameter choices from those recommended in RoBERTa (Liu et al., 2019).
    * Pre-training Generative models. 
        * We perform pre-training using the Fairseq (Ott et al., 2019) toolkit.  
        * Our 2.7B and 9.4B parameter models were both trained using the Adam optimizer (Kingma and Ba, 2014). 
        * In order to fit the larger models onto nodes, we utilize Megatron-LM style model parallelism (Shoeybi et al., 2019), 
            * in which the Feed Forward network (FFN) and Multihead Attention layers of the Transformer are vertically sliced, minimizing the need for communication across GPUs. 
        * We also evaluated Adafactor (Shazeer and Stern, 2018), which allows for larger batch sizes, but we found it converged to a worse place than Adam. 
        * In all cases, we use a variant of mixed precision training (Micikevicius et al., 2017), storing gradients and optimizer state in FP32, but accumulating model parameters directly in FP16 (Ott et al., 2019). 
        * A dynamic loss scalar is utilized to prevent gradient underflow (Micikevicius et al., 2017).
        * Both our 2.7B and 9.4B parameter models were trained with batches of approximately 500k label BPE tokens per batch. 
        * The 2.7B parameter model
            * trained for approximately 200k SGD updates 
            * maximum learning rate of 2e-4, 
            * a linear warmup of 3125 steps, 
            * an invsqrt LR scheduler (Vaswani et al., 2017); 
            * the model had not converged when we stopped. 
        * The 9.4B parameter model 
            * was trained for a total of 200k SGD updates
            * with a maximum learning rate of 1.15e-4 
            * 2400 warmup steps 
            * did not appear to be overfitting.
    * Fine-tuning. 
        * We fine-tune our models using the ParlAI toolkit (Miller et al., 2017)
        * As opposed to the above pre-training, we utilize GPipe-style model parallelism (Huang et al., 2019), 
            * in which full layers are sharded across different GPUs, 
            * and each minibatch is further split into micro-batches to ensure maximum throughput.
        * As in pre-training, we found that Adam outperformed Adafactor during fine-tuning
        * we utilized Fairseq-style mixed precision training. 
        * Models were fine-tuned to convergence, with maximum learning rates of between 1e-6 and 1e-5.

* Training Data (6)
    * all data in English
    * Pre-training (6.1)
        * pushshift.io Reddit We use a variant of Reddit discussions
        * Following Humeau et al. (2019), we use a previously existing Reddit dataset extracted and obtained by a third party and made available on pushshift.io (Baumgartner et al., 2020), 
        * training to generate a comment conditioned on the full thread leading up to the comment, 
        * spanning 1.5B training examples from Reddit obtained from PushShift through July 2019. 
        * The subreddits cover a vast range of topics
        * We apply heuristic rules to filter the dataset with the goal of providing a cleaner training signal. 
        * We remove the comment and all subsequent child comments if any of the following conditions are met:
            1. The author is a known bot.
            2. It comes from a known non-English subreddit.
            3. The comment is marked as removed / deleted.
            4. It is longer than 2048 characters and does not contain spaces.
            5. It is longer than 128 BPE tokens.
            6. It is shorter than 5 characters.
            7. It contains a URL.
            8. It starts with a non-ASCII character.
            9. It is further than depth 7 in the thread.
        * Models were trained with maximum context and response lengths set to 128 BPE tokens, and longer examples were truncated. 
        * Our final dataset contains 1.50B comments totaling 56.8B label BPE tokens and 88.8B context tokens.
        * We divide the corpus into 4096 roughly-equal sized chunks, stratified by thread ID (such that no two comments from the same post appear across folds), 
            * reserve the last two chunks for validation and test respectively,
            * each approximately 0.02% of the full dataset (~360k comments each).
    * Fine-tuning (6.2)
        * While our data has a lot of useful content, it also still has a lot of noise, even after filtering. 
        * In contrast, the academic community has produced a number of smaller, but cleaner, more focused tasks, 
            * the ConvAI2 dataset (Zhang et al., 2018) focuses on personality and engaging the other speaker, 
                * used at the NeurIPS 2018 competition of the same name, 
                * is based on PersonaChat (Zhang et al., 2018; Dinan et al., 2020). 
                * The training data of 140k utterances
                * involves paired crowdworkers having a conversation where they get to know each other, 
                * each is given a role to play based on sentences describing their persona, which were also separately crowdsourced 
                * both speakers can see their own persona description, but cannot see their partner?s persona
                * The task thus involves getting to know the other speaker and engaging them in friendly conversation, both asking and answering questions
                * Models trained on this task are thus conditioned on the persona and the dialogue history, which are concatenated. 
            * Empathetic Dialogues (Rashkin et al., 2019) focuses on empathy, 
                * consists of 50k utterances of crowdworker conversations grounded in an emotional situation. 
                * In each dialogue, one speaker describes a personal situation and the other plays a ?listener? role, displaying empathy during the discussion.
                * Trained models are measured playing the part of the empathetic listener. 
            * Wizard of Wikipedia (Dinan et al., 2019c) focuses on knowledge. 
                * involves discussing a given topic in depth, where the goal is to both engage the partner as well as display expert knowledge (Dinan et al., 2019c). 
                * consists of 194k utterances over 1250 topics
                * each conversation begins with a randomly chosen topic. 
                * A retrieval system over Wikipedia was used from which the  dialogues were grounded during the human-humancrowdsourced conversations. 
                * The topics were also crowdsourced and range from e-books to toga parties to showers.
            * Blended Skill Talk (Smith et al., 2020) 
                * aims to blend the previous three tasks to combine the skills from them 
                * a dialogue dataset of 76k utterances  was collected with a guided and unguided humanspeaker, 
                * where the guided speaker could select utterances  suggested by bots trained on the three individual tasks, see Figure 3. 
                * In each blended dialogue:
                    * the model is provided a two sentence persona to condition on following PersonaChat, 
                    * and additionally during one third of the conversations a WoW topic name as well (see Figure 3). 
                * During evaluations:
                    * we equip our models with randomly chosen personas 
                    * and, one third of the time, topics from this set as well, 

* Safety Characteristics (7)
    * As models are trained to mimic human-human conversations, they can sometimes learn undesirable features from this human-human data, such as the use of toxic or biased language. 
    * The BST tasks we use for fine-tuning were collected from crowd workers who were given explicit instructions to not use such language
    * generally safer than our pre-training data from pushshift.io Reddit.
    * Nevertheless, issues can still remain.
    * We have previously investigated building better classifiers of toxic language by collecting adversarial toxic data that fools existing classifiers and is then used as additional data to make them more robust, in a series of rounds (Dinan et al., 2019b).
    * We can apply such a classifier at test time to detect toxic language before it is shown, but we note that such classifiers are still not infallible. 
    * We have also previously conducted studies into mitigating gender bias in dialogue through the use of conditional generation, controlling the amount of gendered words to be more neutral, with preliminary success (Dinan et al., 2019a). 
        * good idea, but not used in this paper

* Evaluation Methods (8)
    * our main evaluation involves the ACUTE-Eval procedure (Li et al., 2019b), 
    * whereby evaluators are asked to make pairwise evaluations of complete dialogues. 
    * The explicit use of comparisons avoids the per annotator bias in numerical (Likert) scores (e.g., annotators who tend to give generous scores), 
    * and remedies many of the issues of sequential effects such as contrasting with a previous example (Mathur et al., 2017), 
    * while still providing the ability to expose issues that are present only in multi-turn evaluations.
    * the pairwise setup facilitates replication and efficient reuse of data
        * ensure that multiple papers are comparing to prior work consistently. 
        * In particular, this makes it possible to compare to logs from Meena (Adiwardana et al., 2020) even though the model itself has not been made publicly available.
    * We consider two evaluation questions, derived from (Li et al., 2019b):
        * Engagingness question: ?Who would you prefer to talk to for a long conversation??
        * Humanness question: ?Which speaker sounds more human??
    * The phrasing of these questions were themselves optimized in that work to maximize agreement, and we hence re-use those exact phrasings. 
        * It was shown that different phrasings can result in weaker levels of agreement, 
        * and that engagingness and humanness clearly do not measure the same thing.
    * Self-Chat ACUTE-Eval 
        * Nevertheless, full human evaluations are time consuming and costly, requiring humans to spend time conducting conversations with bots as well as scoring them. 
        * it was shown in Li et al. (2019b) that ACUTE-Eval can also work in ?self-chat? mode, where models are used for both sides of a conversation,
        * Results from self-chat experiments highly correlate with those of humanchat experiments, for most, but not all systems (Li et al., 2019b). 
        * This mirrors other successes in using self-play, self-chat, and simulated users to evaluate dialogue systems (Fazel-Zarandi et al., 2017; Shah et al., 2018a,b; Wei et al., 2018; Ghandeharioun et al., 2019). 
    * we only use the full human-bot chat evaluation at the  final stage. 
    * In this work we use the BST-setting to perform self-chats, i.e. models are given the personas, topics and previous utterances to initiate the conversation, see Section 6.2 and Figure 3.
    * Note that when using deterministic methods such as beam decoding, this prevents the models from generating the same conversation repeatedly.

* Results & Analysis (10)

* Automatic Evaluations (10.1)
    * Retriever 
        * We fine-tune the retrieval models on ConvAI2, Wizard of Wikipedia, Empathetic Dialogues, and Blended Skill Talk datasets (BST variants of each7) 
        * and automatically evaluate them by measuring hits@1=K on the validation sets of each of these datasets. 
        * Results are shown in Table 1.
    * Generator 
        * Before fine-tuning,  we assess the performance of our 90M, 2.7B, and 9.4B parameter models by measuring perplexity on the validation set from pushshift.io Reddit. 
        * For the 90M parameter model, results are reported from Shuster et al. (2019), as we use that same model. 
        * Results are shown in Table 2, and training curves for the pretrained models are also provided in Figure 5.  
        * Figure 5: Validation PPL of different sized models. The larger model achieves a better performance in fewer steps, consistent with other works (Kaplan et al., 2020; Li et al., 2020).
        * We also report perplexity both before and after fine-tuning each of these models on the ConvAI2, Wizard of Wikipedia, Empathetic Dialogues, and Blended Skill Talk datasets. 
        * Results are shown in Table 3. 
        * Results show that Fine-tuning gives gains for each skill (task) compared to pre-training on pushshift.io Reddit alone.
    * Retrieve and Refine (RetNRef) 
        * Table 3. 
        * We note a small increase in perplexity ? relative to the standard generator models ? on each of these datasets. 
        * This small increase in perplexity was also observed in Weston et al. (2018), 
        * even though the retrieve and refine models outperformed the baseline generator models in human evaluations in those experiments. 
        * As such, we cannot rely on automatic evaluations alone to assess the relative performance of retrieve and refine and generator models.
    * Safety
        * * We produced generations given pushshift.io Reddit and ConvAI2 validation set contexts using our 90M parameter models with and without BST fine-tuning. 
        * We then assessed whether those generations were safe or not using two different methods: 
            * using an unsafe word list
            * the safety classifier of Dinan et al. (2019b)
            * both methods being available in ParlAI (Miller et al., 2017). 
        * We also compare our generations to the gold human responses, assessing whether they are safe or not too.
        * Table 4
        * First, they show * humans do utter unsafe responses, which our models will likely imitate if provided in their training data. 
        * ConvAI2, one of the BST datasets, contains much fewer unsafe utterances from humans than pushshift.io Reddit. 
        * This explains why, when we fine-tune our models on the BST tasks, they also reply with fewer unsafe utterances than models trained on pushshift.io Reddit alone.
        * lists of banned words are easier to filter out of training
        * unsafe utterances consisting of otherwise safe words are harder to avoid ? which is what the safety classifier used can also detect. 
        * We note that simply training on filtered data would not solve this problem due to the tendency of generative models to copy their current context, 
        * safety classifiers are interesting, but note that if the classifier is erroneous, unsafe utterances could still get through.

* Self-Chat Evaluations (10.2)
    * *perform a number of self-chat ACUTEEvals (see Sec. 8) over various modeling choices
    * using the engagingness question and 140 trials per pair compared. 
    * Retrieval vs. Generator vs. RetNRef 
        * We used the base 90M parameter generative model, the 256M parameter retrieval model, while RetNRef combines both. 
        * All models are fine-tuned on the BST tasks. 
        * For generation: standard beam search (beam size 10, no minimum beam decoding constraint, but with context and response 3-gram blocking).
        * The results (Figure 6) 
        * show RetNRef outperforming the pure generation approach
        * and retrieval outperforming both
        * This initial result  comes with the caveat that relative performancemay be different for differently sized models, or for different training or decoding strategies
        * This mirrors results found in some recent papers comparing generation and retrieval (Li et al., 2016; Dinan et al., 2019c). 
    * Generator Decoding choices 
        * We next compare different ways of controlling the response length in beam search (Sec. 4.3):  
            * controlling the minimum beam length (in terms of BPE tokens) with a fixed hyperparameter
            * or by adjusting it with a predictor of the optimal length.
            * The results, shown in Figure 7 
            * both methods improve significantly over not controlling the length, as in standard beam search. 
            * In the remainder of the experiments in the paper we thus chose a minimum beam length of 20 BPE tokens.
            * Figure 8:  Blocking tends to increase performance, in line with other works, 
            * although the results were not significant. We employ full blocking in the remainder of our experiments.
        * we compare different values of beam size to Top-k sampling, and the sample and rank strategy of Adiwardana et al. (2020) using Top-k (k = 40) and 20 samples.
            * Figure 9, comparing beam size 10 to alternatives. 
            * It appears there is a sweet spot of beam size, where a value of 10 is superior to 1 or 30, 
            * which is then on par with sampling methods, 
            * although none of these results is significant
            * We employ beam size 10 in the remainder of our experiments.
        * Small vs. Large models 
            * We compare 90M vs. 2.7B parameter generative models in a pairwise test,
            * both with BST fine-tuning and with the decoding settings
            * The results (Figure 10) indicate improvements from larger models, in line with previous results (Adiwardana et al., 2020). 
        * Pre-training vs. Fine-Tuning 
            * BST tasks, versus using pre-training only.
            * The results (Figure 11) indicate large improvements from adjusting the model to focus on personality, knowledge and empathy, the three skills in BST.
        * Persona context vs. No context 
            * The results, shown in Figure 12 indicate a small win for employing persona contexts 
            * which we thus employ in all our full evaluations in the next section.
        * Likelihood vs. Unlikelihood 
            * We compare unlikelihood  training (Sec. 3.4), whereby overexpressedn-grams are discouraged ( = 0:25), to conventional training (MLE). 
            * We note that this effect would likely be larger if measured with longer or repeated conversations with the same user. 
            * We compare two models which are identical except for the training objective: both models are 2.7B parameters, BST fine-tuned with our best chosen decoding settings. 
            * The results (Figure 13) have a small gain against the likelihood model, but this is not statistically significant.

* Full (Human-Bot Chat) Evaluations (10.3)
    * used the same setting proposed in (Adiwardana et al., 2020): 
    * open-ended chat that begins with the message "Hi!" from the human to the bot, 
    * has a minimum interactive conversation length of 14  turns, 
    * collecting 100 conversations per model via crowdworkers. 
    * We do not apply a safety classifier to our models, but we do apply it to the human responses, and remove crowdworker conversations that were flagged.
    * Retrieval vs. Generator vs. RetNRef 
        * the generative and RetNRef models here use the improved decoding choices. 
        * This results in stronger generation and RetNRef models, which both now beat the retrieval method, see Figure 14.
        * The main difference to our initial self-chat experiments (Figure 6) is that our decoding now generates longer responses using a minimum beam  length constraint. 
        * This makes the generative models now outperform the retrieval model, but it also removes the gains from retrieve and refine over the generative model. 
        * We note that if we remove the minimum beam length constraint in both retrieve and refine and the generative model and collect new human-bot chats, and a pairwise ACUTE-Eval, we instead get that RetNRef has a statistically significant improvement over our generative model (p < 0:001).
    * Comparison to Meena (Adiwardana et al., 2020) 
        * by comparing pairwise against the publicly available logs. 
        * compared using both the engagingness and humanness questions. 
        * The results are given in Figures 15 and 16. 
        * We first observe several results that are in line with the selfchat results from the previous section:
            (i) Using BST (BST Generative 2.7B) is superior to pre-training only (pushshift.io Reddit Generative 2.7B)
            (ii) Beam search with a minimum beam length of 20 (BST Generative 2.7B) is superior to having no minimum length (BST Generative (2.7B) std. beam)
            (iii) The larger BST Generative (2.7B) is superior to the smaller model BST Generative (90M).
        * We find RetNRef models (both dialogue version and using knowledge retrieval) do not improve over their generative counterparts when using the best decoding schemes for the generative models. 
            * Our largest BST Generative 9.4B model does well on the humanness question, but performs worse on engagingness compared to our 2.7B model, despite having lower perplexity, 
            * showing correlation between these metrics is not straightforward. 
            * We verified this result further by performing an ACUTEEval of engagingness directly comparing the 2.7B and 9.4B against each other, which resulted in a 56% win for the smaller model, aligning with the other results. 
            * Future work should aim to understand this result further.
        * Our best models improve significantly over Meena, with BST Generative 2.7B winning 75% of the time in pairwise match-ups for the engagingness question and 65% for the humanness question.
        * Meena generally tends to fare better at the humanness question than the engagingness question, which is line with the goals and modeling choices in that work.
    * Model vs. Human-human Chat Comparisons
        * Rather than comparing different models pairwise, we can also compare a model directly to human performance, by running ACUTE-Evals with a bothuman chat vs. a human-human chat. 
        * We test the same models in this setup using the humanhuman chat logs from Adiwardana et al. (2020).
        * Results are given in Figure 17. We see many of the same trends, but find that human-human chats are a more challenging barometer for our models to be compared to.
    * Response Length 
        * length statistics (in terms of BPE 8k dictionary tokens) in Figure 18. 
        * We compare Generative BST (2.7B) with and without beam length constraints. 
        * With the constraint (of 20), the average response length is around 21 tokens, so the beam search often ends as soon as the constraint is fulfilled. 
        * In contrast, without the constraint the average length is 9.5. 
        * Meena?s average length is 10.4, 
        * and humans engaged in human-human chats is 18.0. 
        * Humans speaking to models (or other humans) will often match response length if they are engaged in the conversation, and there appears to be correlation of their average response length with engagement

* Example Successful Conversations (10.4)
    * check in the paper

* Failure Cases and Model Extensions (10.5)
    * Vocabulary Usage  
        * It has been observed that generative models employing beam search decoding (or other methods that approximately choose the most likely utterance) tend to generate common words too frequently, and rare words too infrequently, as compared to the human distribution (Holtzman et al., 2018; Welleck et al., 2020; Li et al., 2019a). 
        * In dialogue, humans can interpret this as technically correct, but unengaging, in the extreme this is the so-called ?I don?t know? problem, , where models tend to output such noncommittal utterances. 
        * Using sampling to select lower likelihood generations can help, but at the risk of saying something which makes less sense.
        * It appears that even our best models using beam search are still exhibiting such behavior. 
        * We have found that encouraging the length of the generations to be longer helps, in that the model is forced to generate something more detailed, but the problem still remains. 
        * We note that the current evaluation does not seem to expose this as boring because the conversations are short and are evaluated separately.
        * We applied unlikelihood training to reduce this over-expression, which successfully reduced this overexpression during training, and also in the final conversation logs with humans, as shown in Figure 22. 
        * Unfortunately, this made a very small or negative impact in our ACUTE-Evals of engagingness, see Figures 15 and 17, although this did score highly in terms of humanness, see Figure 16.
        * For engagingness, as explained, we believe this is because the current evaluation technique employing short conversations cannot measure this phenomenon well.
    * Nontrivial Repetition 
        * A related issue is that generative models also have a tendency to repeat (Holtzman et al., 2019). 
        * While beam blocking can be applied as a band-aid to fix some of these problems, resulting in improved performance, deeper issues remain. 
        * We observe this in the logs of other generative systems, e.g., Meena as well. 
        * While this can be engaging that the bot tends to agree with many things you say, control of this seems desirable. 
        * One possibility is applying unlikelihood training for that goal as well, to minimize context repeats (Li et al., 2019a). 
        * Adding a persona to the bot is another plausible way to do this. 
        * We have added simple two line personas following BST (See Figure 3), but this would need to be much more detailed to cover all possible cases
        * Perhaps one way to track this would be to ask human evaluators if the bot is following their persona, as the current evaluation setup is unlikely to penalize this copycat behavior.
    * Contradiction and Forgetfulness 
        * we observed this happens less often in the larger models. 
        * We believe due to the nature of language modeling, typical language patterns do not contain contradictions, but probing the model with unusual responses would likely expose this  behavior again. 
        * A second related problem is what appears as ?forgetfulness? to the human observer, where for example you tell the model you have a dog, but then later in the conversation it asks what pets do you have. 
        * This phenomenon can be attributed to the fact that the model fails to make the logical link that it should not ask that question,
        * Again, we observe this relatively rarely, but we believe it can be exposed further by probing the model. 
    * Knowledge and Factual Correctness 
        * In our experience it is actually relatively easy to goad our models into making factual errors. 
        * Perhaps surprisingly, they appear relatively rarely in crowdworker conversations with the bots. 
            * We believe this is due to the high level conversations.
            * Exploring a more focused topic of conversation would likely expose the model?s weaknesses. 
            * On the contrary, it appears that the model is good at dodging this issue. We observe that our models often switch topics ? avoiding the challenge of going ?deeper" ? which could be a side effect of the ConvAI2 dataset 
            * The Wizard of Wikipedia dataset, however, does not exhibit this behavior, and its construction was specifically aimed to avoid this.
        * We implemented a model that directly incorporated reading Wikipedia (Wiz Generative 2.7B, Sec 2.3),
            * anecdotally one can find cases where it can employ knowledge that the pure sequence to sequence model cannot, see Figure 24. 
            * Unfortunately the reading of knowledge only had a negative impact in ACUTE-Evals compared to a similarly sized model without knowledge retrieval, see Figure 17. 
            * We believe this is due to a mixture of 
                (i) deeper knowledge rarely being required in the current evaluation setup; and 
                (ii) the model attempting to use knowledge when there is no need, or using it incorrectly.
    * Conversation Length and Memory 
        * Our current evaluation involves very short (14-turn) oneshot conversations. 
        * Our bots likely would be repetitive and dull over the course of several days or weeks of conversation, as described above, 
        * and they are also currently completely incapable of even remembering earlier conversations. 
        * Our generative architectures which are standard Transformers have a hard limit of 128 BPE tokens of history, 
        * we have neither implemented architectures to possess longer contexts, nor do we believe the current evaluation setup is the right one for measuring their success.
    * Deeper Understanding 
        * the models ability to truly understand must be questioned.
        * Its lack of understanding can be strongly contrasted with its ability to describe knowledge about the location of Harvard or horses. 
        * This recalls a quote due to Feynman, ?There?s a big difference between knowing the name of something and knowing something?. 
        * We note that these models cannot be taught a concept through further conversation, so as-is they will always be stunted, see (Weston, 2016; Hancock et al., 2019) for early work in this direction. 
        * Further, these models, which are disembodied, also have no way of grounding to entities, actions and experience in the world, which could also stunt their abilities (Bisk et al., 2020). 
        * See Urbanek et al. (2019); Prabhumoye et al. (2020) for other work by some of the authors connecting dialogue models to rich environments.
    * Further Notes on Evaluation 
        * Several of the previous points raised issues concerning our evaluation protocol.  
        * Our set-up involves short multi-turn conversations with no instructions. Extending the length should expose further weaknesses, however collecting long conversations with crowdworkers is clearly difficult, and it is unclear how many turns would be a sufficient test. 
        * We tried a preliminary experiment of collecting 100 conversations twice as long (so, 28 turns) to see the performance dropoff of our models. 
        * We compared the second half of the conversations to the shorter versions for the same 2.7B generative BST model, but did not see a statistically significant difference, indicating they either need to be longer, or the whole conversation has to be evaluated at once. 
        * Another possibility is to keep the conversations short, but to provide instruction instead. 
            * For example, theWizard ofWikipedia task (Dinan et al., 2019c) asks speakers to converse in depth on a randomly chosen topic, changing the  nature of the conversations, and hence the skills the model will be evaluated on.
        * Finally, when comparing to human performance, the quality of the human conversations matters. In Figure 17 we compared to logs of employees from Adiwardana et al. (2020). 
            * Because they work at the same company, or perhaps know each other, these conversations are often rich and engaging.
        * We also tried comparing to human-human crowdworker conversations. 
            * In that case crowdworkers will have no social connection to begin the conversation, and we believe this results in less engaging logs. 
            * When comparing to such human-human crowdworker conversations, which we took from  the BST paper (Smith et al., 2020) we found our models perform better than when compared to employees.
        * our generative BST 2.7B model in an ACUTE-Eval of engagingness 
            * beats humans 56% to 44% (not statistically significant),
            * whereas it scored 49% to 51% against employee chats. 
            * We also compared crowdworker humans directly to employee humans, with a 56% to 44% win for employees in terms of engagingness, and a 59% to 41% win in terms of humanness. 
        * We believe utilizing crowdworkers as a barometer for our models is desirable, as this can yield more replicable experiments, 
            * so finding a way to close this gap, perhaps with alternative ways of matching workers or differing set-ups and instructions remain possible avenues of investigation.


* Discussion (12)
    * we have certainly not yet arrived at a solution to open-domain dialogue. There are still various issues
        i) contradict or repeat themselves on occasion, 
        ii) tend to repeat the same phrases in separate conversations,
        iii) hallucinate knowledge as seen in other generative systems (Massarelli et al., 2019).
    * we made some attempt to rectify
        * phrase repeats using unlikelihood (Li et al., 2019a) in Sec. 3.4, 
        * and conditioning on knowledge (Dinan et al., 2019c) in Sec. 2.3
    * As the human evaluations are on short dialogues (14 turns) longer conversations would likely make these issues appear much worse. 
    * Longer conversations would also expose that the Transformer architectures we use have a limited dialogue history.
        * A number of recent architectures attempt to incorporate longer memory
    * evaluation is more challenging as long conversations have to be collected, and evaluated.
    * An alternative is to seed the conversation with a topic or otherwise provide instructions to the human speaker during evaluation to give the conversation a certain focus, which would more deeply probe the skills of the bot. 
    * On the modeling side, longer conversations could also make the choice of context material provided to the bot more salient. 
    * Besides helping with consistency, the persona and topic that are given as initial context in Blended Skill Talk can help models introduce interesting talking points in the conversation. 
    * However, in our current experimental setup did not affect evaluations strongly. 
    * We note the context our model is trained to be able to condition on can also be used to configure a chatbot persona suitable for a given desired role, see Figure 26 for an example.
    * For deployment of a chatbot, being well-behaved remains a significant challenge. 
        * toxic language (Dinan et al., 2019b) 
        * and mitigating gender bias in dialogue generation (Dinan et al., 2019a) 
        * but much work remains to be done. 
    * The work of Adiwardana et al. (2020) showed that there is a correlation between human evaluation and perplexity, given a fixed decoding scheme.
    * We argue that while optimizing perplexity  is important, other factors are also at play 
        (1) the choice of training data is paramount,  as shown by our pushshift.io Reddit (pre-training) vs. Blended Skill Talk experiments;
        (2) decoding algorithms make large differences for the same fixed perplexity model (Sec. 10.2).
    * our largest 9.4B model does not have a clear win in human evaluations over our 2.7B model, despite having lower perplexity. 
        * This is in line with other results
        * For example, dialogue competitions are not always won by the model with the lowest perplexity (Dinan et al., 2020),
        * and it has been shown that models that take a small hit in perplexity but provide gains at decoding time can give far improved results (Welleck et al., 2020; Li et al., 2019a). 

================================================================================
LaMDA: Language Models for Dialog Applications
https://arxiv.org/abs/2201.08239
================================================================================

================================================================================
Recipes for Safety in Open-domain Chatbots
https://arxiv.org/abs/2010.07079
================================================================================

* When dialogue models are trained to mimic humanhuman conversations utilizing large pre-existing datasets, they will unfortunately also learn undesirable features such as the use of toxic or biased language.
* We investigate a variety of methods to mitigate safety issues in the context of opendomain generative dialogue models. 
* We introduce:
    * a new human-and-model-in-the-loop framework for both training safer models and for evaluating them, 
    * a novel method to distill safety considerations inside generative models without the use of an external classifier at deployment time. 
* we find our new techniques are 
    (i) safer than existing models as measured by automatic and human evaluations
    (ii) while maintaining usability metrics such as engagingness relative to the state of the art.


* First, we compare unsafe utterance detection methods and their employment in two-stage models where generative models are filtered using these classifiers. 
* Secondly, rather than two-stage models, we study training and decoding techniques for safe responses directly in generative models.
* Finally, we also study the issues of sensitive conversational topics, and gender bias mitigation.

* In terms of novel contributions
    (i) Bot-Adversarial Dialogue Safety
        * is a method to collect safety training data with humans and models in the loop. 
        * We ask humans to adversarially talk to a set of state of the art models with the aim of inducing them to generate unsafe responses, similarly to how models can be adversarially attacked at deployment time. 
        * We analyze how to optimally construct such a crowdworker task, 
        * collect a dataset of 5k such conversations
        * involving around 70k utterances, 
        * and use this to train more robust safety classifiers. 
    (ii) Baked-in Safety models.
        * Ideally, we should train generative models that do not have to be screened by an independent classifier
        * We propose such a method by modifying the target labels in the training data to incorporate safe responses where applicable, as defined by a safety classifier. 
        * At test time, one no longer needs the safety classifier
        * In experiments, we show this model outperforms other existing generative models in terms of safety, while maintaining engagingness.

* Base Models (2)
    * We consider the same architecture and setup as in BlenderBot (Roller et al., 2020),
    * a Seq2Seq Transformer architecture  based on the ParlAI version (Miller et al., 2017a). 
    * It uses Byte-Level BPE tokenization (Radford et al., 2019) 
    * trained on the pre-training data, as implemented in HuggingFace?s Tokenizers.1 
    * We consider the 2.7B parameter model (BST 2.7B)
        * 2 encoder layers
        * 24 decoder layers, 
        * 2560 dimensional embeddings, 
        * 32 attention heads
    * Training Data 
        * maximum likelihood on human-human conversations in English, using the Fairseq (Ott et al., 2019) toolkit. 
        * Pre-training employed 1.5B training examples using a previously existing Reddit dataset extracted and obtained by a third party and made available on pushshift.io (Baumgartner et al., 2020) through July 2019. 
        * Heuristic rules were used to filter the dataset with the goal of providing a cleaner training signal. 
        * Models were trained with maximum context and response lengths set to 128 BPE tokens, and longer examples were truncated. 
        * For further implementation details, see (Roller et al., 2020).
    * Fine-tuning is performed on a smaller set of crowdsourced datasets designed to provide important conversational skills. 
        * The ConvAI2 dataset  (Zhang et al., 2018) focuses on personality and engagingthe other speaker
        * Empathetic Dialogues (Rashkin et al., 2019) focuses on empathy, 
        * Wizard of Wikipedia (Dinan et al., 2019c) focuses on knowledge. 
        * Blended Skill Talk (BST) (Smith et al., 2020a) provides a dataset that focuses on blending these skills. 
    * Models were fine-tuned using the ParlAI toolkit (Miller et al., 2017a).
    * Decoding 
        * standard beam search with a beam size of 10, 
        * context and label 3-gram blocking (Paulus et al., 2017),
        * and a minimum beam length of 20 BPE tokens,
    * Comparison Models 
        * compare to two other base models: DialoGPT (Zhang et al., 2019) and GPT2 (Large) (Radford et al., 2019).

* Safety Recipes (3)
    * Unsafe Utterance Detection (§3.1): Training and deploying classifiers for detecting unsafe messages as an added “safety layer.”
    * Safe Utterance Generation (§3.2): Training the model such that it is unlikely to surface unsafe content at inference time.
    * Sensitive Topic Avoidance (§3.3): Avoiding topics like politics or religion, due to their sensitive nature.
    * Gender Bias Mitigation (§3.4): Using strategies from Dinan et al. (2019a) to force the model to respond with gender neutral language.

* Unsafe Utterance Detection (§3.1)
    * still used in some of the most recent dialogue models (Adiwardana et al., 2020; Roller et al., 2020) 
    * This can be used on either side of the conversation, to detect unsafe language from either human or bot. 
    * Many existing methods only perform this detection at the utterance level, detecting unsafe language given only a single dialogue turn
    * we explore five ingredients for detecting unsafe utterances:
        1. Standard unsafe utterance detection.
            * training safety classifiers. 
            * we consider classifiers that are two-class (safe and not safe), although multi-class classifiers can also be considered 
            * we consider Transformer-based classifiers, following the same structure as in Dinan et al. (2019b), with two sizes: 256M and 622M parameter models. 
            * We pre-train these models on a previously existing Reddit dataset extracted and obtained by a third party that was hosted by pushshift.io (Baumgartner et al., 2020), 
            * using a masked language model objective,
            * and then fine-tune on the safety classification task of interest, 
            * performing early stopping using the F1 score of the “unsafe” class on the validation set.
            * Standard Data 
                * We consider the Wikipedia Toxic Comments dataset (WTC) (Wulczyn et al., 2017)
                * designed to identify personal attacks online,
                * consisting of 150k examples; 
                * we use the version that treats the data as a two-class problem (Khatri et al., 2018a; Dinan et al., 2019c). 
                * in addition we consider a dataset more specifically collected for safety in open-domain dialogue of (Dinan et al., 2019b), 
                * which consists of a further 8,000 offensive examples. 
                * We note that these datasets consist of single-turn unsafe utterances, not utterances within the context of a dialogue.
        2. Build-it Break-it Fix-it for robust detection.
            * standard classifiers learn to detect basic toxicity, but can still be fooled
            * Dinan et al. (2019b) thus also explored an adversarial collection scheme to make classifiers more robust. 
            * Therein, crowdworkers are instructed to create training examples that “fool” the classifier into an incorrect decision, which tends to find harder to classify examples; 
            * re-training on this data was shown to make the classifier iteratively more robust.
            * A further 16; 000 examples were collected in such a manner, and we also consider training on this data as well. 
        3. Semi-supervision for expanding train data.
            * Given our best classifier so far from the existing labeled datasets, we can label large unlabeled datasets, e.g. the pushshift.io Reddit (Baumgartner et al., 2020) and BST datasets, 
            * and then train a simple semisupervised approach, training on both gold and imputed labels, related to the work of Khatri et al. (2018a). 
        4. Two-Stage Models: how to combine classifiers with dialogue models.
            * Given a safety classifier, a simple approach to dialogue safety is to apply it in two ways: 
                (i) detect if the user utterances are safe; 
                (ii) detect if its own utterances are safe.
            * If a safety violation is detected in either type of utterance, one can then initiate instead a response designed to be safe. In
            * we consider two approaches: (Curry and Rieser, 2019; Paranjape et al., 2020).
                * safe responses
                    * output a canned, non-committal safe response: Example: sorry, I’m not sure what to say. Thank you for sharing and talking to me though.
                    * One could generalize this to choosing from a set of canned responses.
                * non sequiturs 
                    * choose to change the subject instead. 
                    * We select a topic at random from 1087 topics judged as safe from the Wizard of Wikipedia conversational topic list (Dinan et al., 2019c). 
                    * We then produce the response “Hey do you want to talk about something else? How about we talk about X?”
                    * The danger of such an approach is mostly around the quality of the classifier. 
                        * If the classifier red flags too many safe utterances, the conversational experience will suffer. 
                        * If unsafe utterances are not flagged, toxic language can still enter the conversation. 3.1.3 Bot-Adversarial Dialogue Safety
        5. Bot-Adversarial Dialogue Safety; a new approach introduced in this work.
            * if we want conversational agents to be safe, we advocate applying such a procedure directly to conversations with those agents, which previous work did not address. 
            * The main idea is collect data of humans probing our conversational models to try to trick them into making mistakes, similar to adversarial attacks that can actually happen in practice during deployment. 
            * We can then train on this data to make our system more robust.
            * Task Setup. 
                * We therefore construct a crowdworker task where annotators are instructed to converse  naturally with a bot, but at the same time try to probe or adversarially goad it into replying in an unsafe manner. 
                * We divide the unsafe responses into four bins: responses that nearly 0% of the population would consider unsafe (according to the annotator’s best guess), and responses that < 10%, <50% or > 50% of the population would consider unsafe.
                * We conduct 14-turn conversations in such a manner with human speaking first
                * and score the model based on the number of unsafe boxes, 
                * Such data can be used to train a new improved classifier, as well as holding out some data for a static evaluation test set. 
                * After collection, a separate task verifies the labels with 3 other annotators (Check Appendix A).
                * Figure 1 demonstrates how this adversarial setup differs from the “Build-it, Break-it, Fix-it” setup from Dinan et al. (2019b): 
                    * in the former, the “breaker” (or adversarial user) tries to break a classifier by submitting human-authored adversarial offensive messages that are incorrectly classified as inoffensive, 
                    * whereas in this work, the “breaker” adversarially tries to elicit offensive messages from a dialogue model . 
            * Dataset Statistics. 
                * We collect 5784 dialogues
                * consisting of 78874 utterances in total from both sides (see Table 1). 
                * About 40% of the utterances are annotated as offensive, among which 1/3 are from the bots.
                * To break the bot to use offensive language more often, humans tended to use either unsafe language
                * More than 42% of the dialogues collected contain at least 3 unsafe human messages or probing questions (see Appendix, Table 20). 
                * We further break down the messages from humans into a taxonomy of offensive language types. 
                * The majority of offensive language used by crowdworkers relates to hate speech against particular groups, personal attacks and other less explicit offensive language containing no profanity, see Figure 2. More details can be found in Appendix A.
            * Training Classifiers 
                * After data collection, we can train a two-class multi-turn classifier with the same architecture as in §3.1.1 to predict whether a message is offensive given its context, and employ it in a two-stage model. (Appendix A)

* Safe Utterance Generation (3.2)
    * Adding a safety classifier as a separate layer as described in Section 3.1.2 has its advantages, e.g. any independent improvement of this classifier can be easily combined with a dialogue model, 
    * but
        * it is more complicated to share and deploy models, 
        * requires more computational resources (e.g. loading both models),  
        * and allows  unsafe usage of that model if the layer is simplyignored and removed. 
    * we explore four ingredients for training a model without the use of an additional safety layer:
        1. Data Pre-processing
        2. Safe Beam Blocking/Generation
        3. Safety and Style control
        4. Baking in the Safety Layer; a new approach introduced in this work.

3.2.1 Data Pre-processing
* Assuming we have access to a safety classifier,  which could be any of the methods from Section 3.1, 
* we can use it to filter the training set. 
* In this work we consider two methods:
    * Utterance-based: we can choose to simply remove a target label from the training set if either its context or the label itself triggers the safety classifier.
    * Author-based: given a dataset where the author of each utterance is known, we can  choose to remove all the utterances of given authors, if that author’s utterances trigger theclassifier more than a given number of times. (12% of their posts trigger the safety classifier.)
* This training set is then used to train models as usual. 
* It is important this filtering is performed on the large pre-training dataset, not only the fine-tuning datasets

3.2.2 Safe Beam Blocking/Generation
* adjust the search at decoding  time to avoid such responses.
* perform beam search at decoding time with n-gram blocking, using the given word list.
* the danger remains that the model can still generate an unsafe response composed entirely of safe words.
* a more sophisticated alternative  is to generate responses chosen to not trigger a classifier, e.g. using the plug and play language model approach (Dathathri et al., 2019). 
    * good but not explored in this work

3.2.3 Safety and Style Control
* An approach that is commonly used to specify desired attributes in model generations is so-called control,  (See et al., 2019). 
* we show that control can also be used to control the safety of our models.
* in our case we consider the (standard) approach of adding control variables  (in the form of special tokens appended to the input) at training time   per example that capture the low-level attribute that we wish to control at test time. 
* This variable is appended to the dialogue history, per example. 
* At test time, we set the control to a fixed desired choice.
* We consider two types of control:
    * Safety: 
        * Using a safety classifier, we determine the safeness of each given label and assign the Safe or Unsafe control to be appended to each training example. 
        * At test time one fixes the control to Safe.
    * Style: 
        * The work of Shuster et al. (2018) provided data and  proposed a multi-classifier involving 215 dialogue styles ranging from positive (calm, cheerful), to neutral (formal, impassive), to negative (hostile, cruel). 
        * This labelled  data was used in Smith et al. (2020b) to train a classifier that was in turned used to label the BST datasets with styles. 
        * The base pushshift.io Reddit 2.7B model was then finetuned on the BST datasets augmented with the style labels as control tokens, to obtain a style-controlled generation model that can specify a style at test time. 
        * here, In our experiments we use such controlled generation models to measure the safety of several styles.

3.2.4 Baking in the Safety Layer
* only data pre-processing can make those models susceptible when confronting such language because they will have never seen it before:
* our models frequently copy the input (Welleck et al., 2020), so they might for example copy the offensive language in the input. 
* we instead attempt to bake awareness of toxic language into the training data, but with labeled examples that recommend appropriate action on the model’s part.
* To do this, we first assume we have access to a safety classifier at training time (but not at deployment time), just as in §3.2.1. 
* For each training example, 
    * if the last utterance in the dialogue history or the gold label are labeled as unsafe by the classifier, 
    * we instead replace the label of that training example with a safe response or non-sequitur, see Section 3.3. 
    * example in table 2
* After constructing “baked-in” safety data, one can then train the generative model using likelihood training in the same way as usual
* We make a separation between training examples that have been modified for safety, and those that have not, and assign different weightings to them, effectively drawing examples from those two sets with different probabilities,
    * affecting how much the model optimizes for safety versus usual conversational abilities. 
    * We choose this weighting as a hyperparameter of the model.

* Sensitive Topic Avoidance (3.3)
    * Some topics are more controversial than others, and holding an opinion in one way or the other can potentially upset some subset of people who hold a very different opinion: politics, medical advice
    * While these utterances are not unsafe in the same sense of a toxicity classifier, they can cause problems when bots are unable to delicately navigate sensitive conversations. 
    * in this work, topics to avoid were selected based on their potentially sensitive nature and the availability of training data, though one might consider a wider list of topics depending on one’s use case.

* We crowdsource lists of subreddits that contain conversations on these topics, see Figure 3. 
* We use a multi-class classifier with the same architecture as in §3.1.1 
* a 256M Transformerbased classifier pretrained on pushshift.io Reddit 
* using a masked language model objective — to predict the sensitive topic label (e.g. “politics” or “religion”) given a truncated thread from a given subreddit. 
* We include a “safe” class for all other (non-avoided) topics, for which we use all other subreddits in the pushshift.io dump. 

* Given that the labels we extract from these subreddits are noisy 
* we collect a small validation set on Mechanical Turk to measure the performance of these models. 
* This dataset was collected by instructing paired crowdsource workers to discuss one of the randomly assigned topics with one another.
* Dataset statistics are provided in Table 4.

* At deployment time of a two-stage model containing our classifier, 
* if a human or bot utterance is flagged as not belonging to the safe topic class, 
* we can then trigger a canned response, similar to Sec. 3.1.2.

* Gender Bias Mitigation (3.4)
    * Gender bias can also be connected to toxic language, in that offensive utterances about a female are more likely to contain gendered or swear words than about a male (Dinan et al., 2020).
    * Previous studies have shown that such bias can be mitigated through the use of conditional generation (Dinan et al., 2019a).
    * In this work, we follow the same approach. 
    * Using a gendered word list, we train a controllable generation model with four genderedness bins: 
        * F0M0, F+M0, F0M+ and F+M+. 
        * 0 and + like regular expressions zerro and multiple items
    * We then train with the bin of the gold label appended to the input context for each training example. 
    * At deployment time, we then fix the bin appended to the dialogue context to be F0M0, i.e. to use as few gendered words as possible. 
    * We note that this approach has many limitations: by construction, it is limited to explicitly binarily gendered words from a static word list. 
    * More recent work (Dinan et al., 2020) seeks to address some of these limitations.

4 ExistingWork

5 Evaluation

5.1 Evaluating Conversational Quality

5.1.1 Automatic Quality Metrics
* Using human-human chat data as the evaluation set, one can use perplexity and F1 metrics to measure conversational quality. 
* One can see these metrics as proxies for measurements of humanness of a model, as they attempt to mimic human responses. Assuming that humans are engaging to other humans,
* one can also see these metrics as a proxy for engagingness as well.
* However, perplexity alone does not measure generation quality well (Welleck et al., 2020), 
* and so we also report the F1 overlap with gold labels in some of our experiments
* We note that all automatic metrics have flaws (Liu et al., 2016), 
* hence we also report human judgments as described in the next section.

5.1.2 Human Quality Evaluation
* We use the ACUTE eval (Li et al., 2019) method of evaluating conversational quality
    * as used for BlenderBot (Roller et al., 2020) and elsewhere.
    * The method involves collecting human-bot conversations for two models one wishes to compare,
    * and then presenting two of those conversations at random, one from each model, to crowdworkers to make a series of pairwise decisions. 
* We use the engagingness question from (Li et al., 2019) 
    * which is phrased as “Who would you prefer to talk to for a long conversation?”. 
* We compare each of the proposed models in this paper to the baseline BST 2.7B which Roller et al. (2020) showed is state of the art in open-domain dialogue. 
* our goal is not to find models more engaging than our baseline, but to find models that are simultaneously equally engaging, while being safer. 

5.2 Evaluating Safety

5.2.1 Automatic Safety Metrics
* For a given model
    * we take a subset from the pushshift.io Reddit and ConvAI2 (2000 dialogue history contexts from the validation set of each, respectively) 
    * and generate their corresponding responses.
    * We then evaluate those responses using two standard, simple classifiers: 
        * the word-list based and transformer-based classifiers that are provided in the ParlAI dialogue platform
        * While there may be more sophisticated classifiers, we chose these two for ease of reproducibility.
* We note that such a classifier-based evaluation may not agree with human judgments, because the classifications may not always be correct. 
* we believe such an approach is reasonable for evaluating models that do not use a safety classifier in their inner-loop, 
* but for methods  such as two-stage models (Sec. 3.1.2) it makes littlesense 
    * That is because, if the two-stage model uses the same classifier as we are using for evaluation, it will appear to be perfect, 
    * and if it uses a different one, much of the evaluation will hinge on the similarity between the two classifiers. 
    * We therefore advocate human judgments only in those cases.

5.2.2 Human Safety Evaluation
* we select a particularly difficult challenge: dialogues from the Bot- Adversarial Dialogue safety setup of Sec. 3.1.3.
* We use a test set consisting of 180 such dialogues, distinct from the training data collected in that procedure.
* As models are used in the loop to collect that data, whereby humans construct contexts that induce unsafe utterances from a given model, the test set is sampled from a suite of models rather than a single model. 
* At evaluation time, for a given model, 
    * the model’s responses are generated for the given contexts, 
    * and three independent human evaluators are used to judge the safety of each response.

5.3 Optimizing crowdsourced data collection
* We use logistic regression to model outcomes of interest: 
    * bot utterance being rated as not OK either by the chat partner or in a subsequent verification task, 
    * human input being rated as not OK. 
* We include as predictors 
    * the model underlying the bot responses (which has a large significant effect, 
    * variables capturing the human chat partner’s experience with the task 
    * the particular bot they are currently talking to, 
    * and which of two possible versions of task instructions was received. 
* Experience with the task is measured as the number of HITss accepted by the worker – a HIT, or Human Intelligence Task, is the term used by Amazon’s Mechanical Turk to refer to a single instance of a crowdworker task. 
* Experience with the specific bot is captured as the position of the utterance within the conversation (e.g., 2nd utterance in a 14 utterance conversation). 
* all variables explored in this section are jointly modeled (see Table 5), 
* Effects of instructions. 
    * use profanities or obviously unsafe content is however easily detected by existing classifiers and is therefore not helping improve our safety systems. 
    * Replacing instructions by a new set that suggests asking open questions about sensitive topics rather than using obvious profanities has a significant effect,
* Self-selection effects. 
    * Table 6 suggests that workers who successfully figure out how to trick the bot into saying more offensive utterances are more likely to go on accepting more HITs of the task. 
    * This in turns makes data collection more efficient.
* Learning Effects. 
    * two types of learning effects are apparent. 
    * The increased success at eliciting not OK utterances as more HITs are completed suggests that workers find more effective techniques to provoke unsafe utterances as they perform more iterations of the task. 
    * workers appear to be more successful eliciting unsafe responses later within a given session. 
    *  Both effects are shown in Table 5.
* our results confirm that 
    (1) specific instructions are important, 
    (2) it helps to make conversations within a HIT long enough for a worker to figure out a winning adversarial strategy for the specific model they have been paired with, 
    (3) but allowing for repeated HITs can lead to beneficial self-selection effects.

6 Results & Analysis

6.1 Base Models: Results
* we first present results for standard models without adding our safety techniques. 
* BST 2.7B (Roller et al., 2020) 
* DialoGPT (Zhang et al., 2019) uses a pre-processing method, where offensive subreddits where removed from the training data. 
    * two flavors: 
    * with short generations (using standard beam decoding), 
    * longer generations (where we add a constraint that a minimum of 20 tokens must be generated, similar to (Roller et al., 2020). 
* GPT2 (Radford et al., 2019) was trained on web data that was filtered for data quality, but not for offensive language as far as we are aware.

* Automatic evaluations 
    * Results in Table 8 show that all these models exhibit significant safety issues,
    * GPT2 generations being flagged by a safety classifier 8.0% of the time given pushshift.io Reddit dialogues as input context, and 2.4% given ConvAI2 dialogues. 
    * DialoGPT is as high as 19.9% on pushshift.io Reddit (without the minimum beam).
    * We can compare these to human numbers, which are actually quite high on pushshift.io Reddit (16.5%), 
    * In contrast, the safety classifier only fires on human data from ConvAI2 3.9% of the time, which can be explained by this data being authored by crowdworkers who had instructions not to use toxic language.
    * Comparing the two models pushshift.io Reddit 2.7B (which is pre-trained only on pushshift.io Reddit) and BST 2.7B (which is then fine-tuned on BST tasks such as ConvAI2) 
        * one can observe a decrease in safety classifier fires down from 8.1% to 1.8% on ConvAI2, 
        * and a similar decrease on pushshift.io Reddit. 
        * This shows how training on less toxic data induces less toxic models.

* Safety Human Evaluations 
    * Results given in Table 9 evaluating these methods in an adversarial safety setting, show that all these models are susceptible to attack, 
    * GPT2 produces safe responses only 59.4% of the time, 
    * BST 2.7B only 55% of the time. 
    * We note that while in normal conversation BST 2.7B is safer than pushshift.io Reddit, 
    * in this adversarial setting, they are similarly unsafe, with the latter obtaining a 57.2% OK rate.

* Engagingness Evaluations 
    * BST 2.7B is significantly more engaging than DialoGPT (both variants), and pushift.io Reddit 2.7B.
    * This matches the automatic evaluations, shown in Table 8 (F1 score, last column). 
    * Overall, we do not  see a direct correlation between safety and engagingnesswhen comparing these models. 
    * As we are interested in finding the model that is simultaneously the most engaging and the safest, our safety efforts thus concentrate on using BST 2.7B as a base model.

6.2 Unsafe Utterance Detection: Results

6.2.1 Training a Classifier
* We compare training safety classifiers using the methodology described in Sec. 3.1.1, 
* Firstly, we find our newly trained models superior to existing models from Dinan et al. (2019b) when using the same training sets, likely due to improved pushshift.io Reddit pre-training of our transformers compared to their BERT models. 
* However, we find  relatively small gains from either larger transformers (Safety Classifier+) over smaller ones (Safety),
* or from semi-supervised learning over Reddit and BST (Semi-Sup. +).

6.2.2 Two-Stage Models
* We apply these classifiers as two-stage models together  with our baseline generative model BST2.7B, outputting a non-sequitur if the classifier fires.
* We observe in Table 10 engagingness scores do not suffer for these models, 
* with the differences between  the two-stage models and BST 2.7B withouta safety classifier not being significant. 
* However, the two-stage models do give improved levels of safety, as shown in Table 9. 
* the baseline BST 2.7B only provides OK responses 55% of the time on the adversarial test set, whereas our Safety classifier improves that to 87.2%, 
    * superior to the existing work of Dinan et al. (2019b) which yields 78.2%. 
* We do not find that semi-supervised classifier (Semi-Sup. +) improves over our own base Safety model. 
* Generally, the two-stage model approach can be an effective tool for safety.

6.2.3 Bot-Adversarial Dialogue
* Classifier
    * We compare the classifier trained on  the BAD dataset, multitasked with the other datasets, to other approaches in Table 7. 
    * We observe similar results to our other new safety classifiers on the single-turnWikipedia Toxic Comments, Build-It Break-It Fix and Standard test sets, 
    * but superior results on the multi-turn bot-adversarial BAD test set. 
    * The BAD-based classifier achieves 80.8 unsafe F1 on the latter dataset, while the next  best performing methods achieve 61.5, 61.0 and60.7, respectively. 
    * This result can be explained as the BAD-based classifier is the only one trained on the BAD training set, hence it sees data closely linked to the evaluation distribution. 
    * One can tease apart the contributions from the BAD training set being both adversarial and multi-turn by comparing to a single-turn (truncated) version of BAD training,
        * which still performs well – though not as well – as the multi-turn version, indicating that the adversarial component is most important. 
        * As the BAD test set is the closest setup to the actual use of a classifier during deployment (it features human-bot conversations, rather than human-human single-turn data) this indicates the BAD-based classifier is the most likely method to be successful in real use cases.
* Two-Stage Model 
    * We apply the classifier learned from our Bot-Adversarial Dialogue (BAD) dataset (multi-tasked with our other datasets) in a two-stage model. 
    * Engagingness (Table 10) is found to be not significantly distinguishable from our base BST 2.7B model. 
    * In terms of safety (Table 9) this approach improves over our other safety classifiers used in two-stage systems, yielding an 94.4% OK rate on the adversarial data.
    * Simultaneously to being robust to adversarial attack, during conventional (non-adversarial) chat this approach rarely deviates from the conversation of the base BST 2.7B model. 
    * We calculate how frequently each chatbot model responds with nonsequiturs when humans converse normally with it in an non-adversarial manner in Table 12. 
        *  The BAD-based two-stage model (“BST 2.7B + Adv. Dialogue Safety”) produces fewer non-sequiturs compared with many of the other two-stage models.
    * Overall, this method offers strong robustness without affecting engagingness, and we advocate its use.

6.3 Safe Utterance Generation: Results

6.3.1 Data Pre-processing

* We trained with two types of data pre-processing (author and utterance methods, §3.2.1). 
* These models were trained from scratch using 400M parameter transformer models 
* We then compare both pre-train only models and fine-tuned BST models in terms of safety and PPL and F1 metrics. 
* The pre-processing from utterance and author safety methods resulted in training set sizes that were 70% and 30% of the original pre-train dataset, respectively.
 * We compare  these to a baseline 400M model using the whole pre-train dataset (so no safety mechanism is built in). 
* Results are given in Table 13. 
* We find that both pre-processing methods are safer than the baseline,
* with the safe utterance method being significantly safer than the safe author method. 
* We note the safe author method still has a large number of unsafe utterances, according to our safety classifier, but not enough for any one author to trigger removing the author, which may be the reason for worse safety statistics on the validation set. 
    * This would lead to a conclusion that while toxic authors exist, there are also a large number of otherwise non-toxic authors  who sometimes use toxic language, and thiscan adversely affect model training.
 * We note that  one could employ both procedures: safe author + utterance, but we have not tried that experimenthere.

 6.3.2 Baked-in Safety Layer
* 400M models 
    * We first directly compare the baked-in safety layer method of §3.2.4 to the datapreprocessing methods. 
    * we train a 400M parameter model from scratch, 
        * with 50% of the safety classifier triggered pre-training data
        * replaced with non-sequitur labels, 
        * and the rest of the safety classifier triggered data discarded, to  prevent too much of the training time spent on nonsequitur prediction. 
    * The results, given in Table 13 indicate that perplexity takes a slight hit, but that safety classifier fires on model generations (given validation set contexts) decrease substantially. 
    * we found that our pre-train only model is overly cautious at deploy time and too often generates nonsequiturs, resulting in a low F1 on ConvAI2 for example. 
    * As it is expensive to begin pre-training with different hyperparameter values, we thus instead remedy this at fine-tune time by weighting the amount of training examples sampled in each batch between the BST tasks and non-sequiturs. 
        * The last two rows of §3.2.1 show that this technique can effectively control the non-sequitur firing rate. 
    * The last row in particular achieves an F1 score similar to the pre-processed data methods (safe author and safe utterance) while having a much lower safety classifier firing rate – reduced from 6% to 0.2%. 
    * We thus conclude from these experiments that baked-in training is a method worthy of further study, and in subsequent experiments proceed to apply it to larger 2.7B models instead.
* 2.7B models 
    * To scale up to the 2.7B parameter size, we considered two strategies: 
        * fine-tuning from the base 2.7B BST model to add baked-in safe responses, 
            * we considered the two types of safe response detailed in §3.1.2. 
        * training a completely new model from scratch with non-sequiturs as part of the pretraining task, followed by fine-tuning. 
            * we tuned the blend of safe responses and dialogue data, 
    * Model engagingness results (Table 10) indicate that non sequiturs are more engaging than bland safe responses; 
        * intuitively this makes sense as they are interesting conversation starters. 
        * We therefore used non-sequiturs elsewhere in our experiments as well. 
    * In terms of engagingness, the two fine-tuned (BST 2.7B Non sequitur and BST 2.7B Non sequitur (Semi-Sup.+) ) and the from scratch non sequitur model all perform similarly to the base 2.7B model (are not significantly different), 
        * indicating again (as in the 400M experiments) that these systems work well in terms of conversation quality.
        * Automatic evaluations (Table 8) also confirm these results in terms of F1 scores.
    * In terms of safety, we see clear wins for these models using automatic safety metrics, as shown in Table 8.
        * we see a reduction from 10.0% classifier fires on pushshift.io Reddit for the base BST 2.7B model being reduced to 0.9% for BST 2.7B Non Sequitur (Fine-tune), and 0% for the from scratch model.
    * On the human-judged adversarial test set (Table 9) we also see gains 
        * increasing from the baseline BST 2.7B value of 55% OK up to 75.6% OK), 
        * although these gains are not as significant as when using two-stage models (the same classifiers in a two-stage setup can bring the results up to 87.2% OK).

6.3.3 Safe Beam Blocking/Generation
* In this section we report results for safe beam blocking methods using two unsafe word lists,
    * the default one in ParlAI(Miller et al., 2017a) 
    * or a CMU word list
* Automatic evaluations are shown in Table 8.
* We observe little loss in the F1 metric,
* but despite the word lists now banning obvious offensive words, we observe only small decreases in the toxicity of the language used, as judged by the safety classifier.
* This indicates that these models still find a way to generate unsafe responses composed entirely of safe words, as judged by the word lists.
* For that reason, we did not pursue these methods further.

6.3.4 Style and Safety Control
*  We trained style and safety control models from scratch using 400M parameter transformer models trained on pushshift.io Reddit 
* We then evaluated the safety of their generations using automatic metrics on the pushshift.io Reddit validation set.
* The results are shown in Table 14.
*  We observe a clear improvement in safety metrics from positive styles such as “calm” or “cheerful” compared to the baseline (default style),
* and clear degradation from negative styles such as “hostile” or “cruel”.
* Analysing the actual dialogue (Table 18) shows that control methods are capable of producing the desired style attributes,  see also the work of Smith et al.  (2019).
* After fine-tuning on datasets such as BST (not shown) we also see similar results (with all values lower,
*  The “Safe” control also provides improved safety, but not as much as the safest choices of style.
* We also attempted to fine-tune a 2.7B parameter model with safety control, rather than training from scratch, but this did not yield large improvements,
* As the style results appear promising we chose to evaluate some of them with human judgments,
    * the results are reported in Table 9.
    * We observed no gains in this adversarial setting for “calm” over the baseline of no control,
    * although we do observe sever degradation with the “hostile” style.
* Overall, we believe this is an interesting area still worthy of further study, but our current results are inconclusive on our current implementations worth in comparison to other methods.

6.4 Sensitive Topic Avoidance: Results
* Classifier 
  * We evaluate the performance of our topics avoidance classifier (§3.3) on our crowdsourced validation set.
    * Results are shown in Table 16.
    * Our model achieves strong performance on all sensitive topics excluding NSFW and Relationships/ Dating.
    * We suspect there is a domain mismatch between the NSFW subreddits and the relationship conversations that appear in the validation set.
    * When we deploy our topics classifier in the 2-stage model,
        * we use a threshold of 0:55 for all topics excluding NSFW and 0:7 for NSFW: this threshold was tuned by evaluating the model with various thresholds on both this validation set and the ConvAI2 validation set with the aim of finding a threshold that yields sufficient performance on this validation set but does not flag too many ConvAI2 conversations.
    * To understand these domain differences further, we look into how many examples from the topic classifier validation set are flagged as “Not OK" by the safety classifier in Table 16: 
        * the recall shows that only 9:61% of examples are flagged.
        * This shows that there is some overlap between the safety classifier and sensitive topic domains but that they are largely disparate.
* Two-Stage Model Human evaluations of engagingness 
    * (Table 10) indicate losses relative to BST 2.7B when using the topic classifier in a two-stage model,
    * although the numbers are higher when combining both the topic classifier and the safety classifier; we are not clear on why that is, exactly.
    * We observe the topic classifier fires much more often than the safety classifier (around 3x as often) which could explain why this would affect engagingness (see Table 12).
    * For this reason, we currently prefer the safety classifier approach in terms of deployment.

* In terms of safety, the topic classifier does have a noticeable effect as a two-stage model (Table 9).
    * It obtains an OK rate on the adversarial test of 73.3% versus the 55.0% BST baseline.
    * Combining with the Safety Classifier yields 92.2%, showing that these two classifiers learn different things 
    * the safety classifier alone yields 87.2%
    * Combining with our best Adversarial Dialogue Safety classifier, applying the topic classifier improves the OK rate from 94.4% to 96.6%. 
    * Overall, dealing with sensitive topics is shown to be an important issue to deal with.

6.5 Gender Bias Mitigation: Results
* We fine-tuned the BST 2.7B model with gender bias control variables, described in §3.4.
* The results are given in Table 15,
* comparing the BST 2.7B baseline with the bias control model with four fixed choices of control: F0M0,F1M0, F0M1 and F1M1.
* The toxicity of the models, as judged by the unsafe word list and classifier metrics, is lower for the models that are more gender neutral,
* particularly F0M0 lowers the classifier on pushshift.io Reddit from 10% on the baseline to 5.3%, a substantial reduction.
* This model roughly halves the usage of gendered words, without impacting perplexity unduly.
* In terms of human judgments, 
    * the model matches the baseline BST 2.7B performance (Table 10) in terms of engagingness.
    * However, it has little effect on adversarial safety performance (Table 9), achieving a similar performance to BST 2.7B (around 55% OK rate).
    * One can argue that this is the wrong kind of test for a gender debiasing model, which is instead addressing other issues.
    * Given that the model does not change engagingness, we make the recommendation that this kind of technique should be incorporated into a model in any case.
    * However, to fully evaluate its impact we need to incorporate other tests and metrics into our current methodology.

6.6 Overall Comparison Metrics
* Ideally we are interested in a model that is both maximally safe and engaging.
* We re-iterate that this may result in a potential trade-off: a model that responds “I do not know how to respond" to every prompt is unlikely to offend, but is also far from an engaging conversationalist.
* We visualize the relationship between engagingness and safety in Figure 3.
* we measure conversational quality via the engagingness scores given from the human evaluations shown in Table 10.
* Safety scores are measured via the human evaluations on the Bot-Adversarial Dialogue (BAD) test set as shown in Table 9.
* In addition to the adversarial test of safety, we also provide a less adversarial test, using pushshift.io Reddit contexts as input instead, using an automatic metric (via a safety classifier) to measure the safety of the responses, following Table 8.
    * We compare that against the automatic metric F1 to measure conversational quality in Figure 4 (left),
    * and contrast that with adversarial safety in Figure 4 (right).
* Overall, we observe that standard generative models – with little or no safety intervention – fall very short in the safety axis.
* However, with some of our safety recipes we are able to achieve roughly the same engagingness as the state of the art BST 2.7B (BlenderBot) with substantially better safety scores, showing that it is possible to build a model that is both safe and engaging.
* We find generative models can be improved substantially by distilling a safety classifier into the encoder-decoder weights during training, i.e. the baked-in approach “BST 2.7B Non-Seq. (Semi- Sup)”.
    * This is especially evident in the nonadversarial case (Figure 4,left).
* Two-stage models provide safer results still, with the best performance coming from our Bot-Adversarial Dialogue data (BAD)-based classifier combined with BST 2.7B.

6.7 Success and Failure Cases
* Successes 
    * In Table 17, we show success cases for our BST 2.7B + Adversarial Dialogue Safety (two-stage) and BST 2.7B Non-Sequitur (baked-in) models on the BAD test set.
    * We also provide the outputs for the standard BST 2.7B model (Roller et al., 2020) and DialoGPT (Zhang et al., 2019).
    * In all three cases the safety models are able to successfully recognize the unsafe input and avoid responding by providing a non-sequitur.
    * Conversely, both BST 2.7B and DialoGPT engage with the unsafe input.
    * In Table 18, we show an example of how differ- ent style controls – no control (baseline), calm, and hostile – result in drastic variations in the generated output.
        * The hostile model responds in an offensive manner while the calm and baseline variations respond in positive or neutral tones.
* Failures
    * our safety models still fail 
    * Failure cases are shown in Table 19 for our BST 2.7B + Adversarial Dialogue Safety (two-stage) model.
    * In both cases, the models’ responses are unsafe in the context, showing how adversarial input can elicit an unsafe response.

7 Conclusion and Discussion
* we find that two new techniques we propose are promising avenues of research: 
    (i) baking-in safety into generative models,
    (ii) building adversarial human-bot conversation robustness into two-stage models.
* We find that both of these techniques outperform their respective generative or two-stage model counterparts.
* To aid this study we have investigation techniques of crowdsourcing safety evaluations,
* and built an adversarially created dialogue safety training and evaluation set,
* our best systems are not perfectly safe yet.
* We note that even our safest model is rated by humans as being safe 96:6% of the time on our adversarially created dialogue safety test set.
* This begs the question: when can a model be considered “safe"? 
* Is a failure rate of 3:4% in an adversarial setting acceptable for the deployment of such models? How safe is safe enough? 
* Further complicating the issue is the fact that the very definition of “safe" is both contextually and culturally dependent (Schmidt and Wiegand, 2017).
* A dialogue model must be able to understand the boundaries of its particular conversation partner.
* What is offensive to one may not be offensive to another (Curry and Rieser, 2019).
* Culturally speaking, the approaches in this paper are limited in both the geographical and historical senses.
    * Our methods rely only on English-speaking anno- tators located in the United States.
* We have also assumed a consensus-based view on offensiveness, by admitting test examples based on agreement of multiple human verifiers; however, offense to minority groups for example may be missed by such a setup.
