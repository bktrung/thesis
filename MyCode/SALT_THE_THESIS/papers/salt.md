Semantic Aware Linear Transfer by Recycling
Pre-trained Language Models for Cross-lingual Transfer
Seungyoon Lee, Seongtae Hong, Hyeonseok Moon, Heuiseok Lim†
Korea University, Republic of Korea
{dltmddbs100, ghdchlwls123, glee889, limhseok}@korea.ac.kr
Abstract
Large Language Models (LLMs) increasingly incorporate multilingual capabilities, fueling the demand to transfer them into target
language-specific models. However, most approaches, which blend the source model’s embedding by replacing the source vocabulary
with the target language-specific vocabulary,
may constrain expressive capacity in the target language since the source model is predominantly trained on English data. In this
paper, we propose Semantic Aware Linear
Transfer (SALT), a novel cross-lingual transfer
technique that recycles embeddings from target
language Pre-trained Language Models (PLMs)
to transmit the deep representational strengths
of PLM-derived embedding to LLMs. SALT
derives unique regression lines based on the
similarity in the overlap of the source and target
vocabularies, to handle each non-overlapping
token’s embedding space. Our extensive experiments show that SALT significantly outperforms other transfer methods and achieves
lower loss with accelerating faster convergence
during language adaptation. Notably, SALT obtains remarkable performance in cross-lingual
understanding setups compared to other methods. Furthermore, we highlight the scalable use
of PLMs to enhance the functionality of contemporary LLMs by conducting experiments
with varying architectures.
1 Introduction
As Large Language Models (LLMs) continue to
demonstrate remarkable performance across various knowledge-based benchmarks and sub-tasks,
the demand for robust linguistic capabilities in multilingual or language-specific contexts has surged
(Wei et al., 2022a,b; Peng et al., 2023; Taori
et al., 2023; Zhou et al., 2024). However, most
multilingual-considered LLMs show limited practicality in target languages due to their English-
†Corresponding authors
centric training and reliance on common vocabulary designed to incorporate a wide range of languages (Puttaparthi et al., 2023; Le Scao et al.,
2023; Lai et al., 2023a; Zhao et al., 2024; Team
et al., 2024; Dubey et al., 2024). This drawback
also correlates with the large portion of their parameters, which are the embeddings of irrelevant
language tokens to target language.
To address this challenge, a promising line
of studies has focused on cross-lingual transfer (Artetxe et al., 2020; Tran, 2020; Gee et al.,
2022; Dobler and De Melo, 2023; Remy et al.,
2024; Liu et al., 2024; Mundra et al., 2024; Ye et al.,
2024). These efforts include replacing a source vocabulary with a target language vocabulary, simply
initializing embeddings with newly manipulated
embeddings stem from source model (Minixhofer
et al., 2022; Dobler and De Melo, 2023; Liu et al.,
2024; Mundra et al., 2024).
However, a newly initialized embedding that relies solely on the source model inevitably experiences limited expressiveness in the target language
due to the immature learning process of the source
model, which does not mainly concentrate on the
target language. Moreover, most approaches adopt
small encoder architectures such as BERT-like
models (Devlin et al., 2019; Liu, 2019; Conneau
et al., 2020) as a source model, leaving their applicability on contemporary decoder-based LLMs
unexplored (Gogoulou et al., 2022; Gee et al., 2022;
Zeng et al., 2023; Dobler and De Melo, 2023; Ye
et al., 2024).
In this work, we propose Semantic Aware
Linear Transfer (SALT), a novel cross-lingual
transfer method that leverages the rich representational power of target language embeddings in
traditional Pre-trained Language Models (PLMs)1
to convert English-centric LLMs into target lan1
In this work, we define PLMs to small-scale language
models with only millions of parameters, developed prior to
the advent of LLMs.
arXiv:2505.10945v2 [cs.CL] 22 May 2025
guage–specialized LLMs. To facilitate the transfer,
we identify the semantically closest tokens for each
non-shared token from a shared vocabulary space.
Based on these semantically similar tokens, we fit
a unique linear regression to transfer the embeddings from the PLM’s space to the LLM’s space.
This procedure enables non-shared vocabulary to
be transferred to LLM space while retaining the
semantic representation embedded in PLM embedding.
In our experiments, we focus on investigating
the potential of transferred embeddings from PLMs
in three aspects: (i) whether the transferred embeddings can be well-aligned with inherent knowledge
in the source model, (ii) the influence on better initialization to the target language and convergence
during continual pre-training, and (iii) the ability of
understanding between target language and mainstream language in cross-lingual environments.
We empirically demonstrate that embeddings
from target language PLMs can be more helpful
for cross-lingual transfer in LLMs than existing
methods. SALT not only surpasses various strong
baselines in downstream tasks but also provides
a better initialization for target language adaptation with faster convergence during continual pretraining. Notably, we find that SALT preserves English capability more effectively while maintaining alignment between English and the target language in a cross-lingual setup. Furthermore, by
conducting additional study on various PLM architectures (Encoder, Decoder, and Encoder-Decoder),
we discover that SALT can be extended to PLMs
with various architectures and can even be a valid
strategy for basic models such as BERT (Devlin
et al., 2019). Our contributions are as follows:
• We propose SALT, a novel embedding transfer method to effectively project the deep representation capabilities of PLM embeddings
onto LLMs based on semantic information.
• We empirically verify that SALT has superiority in cross-lingual environments as well as
downstream tasks and language modeling for
the target language.
• We further show the versatility of SALT in
experiments with various types of PLMs.
• By recycling previously prominent PLMs as
target language embeddings, we demonstrate
the potential scalability and advantage of
PLMs, and propose a new approach to leverage PLMs in the era of LLMs.
2 Related Work
Cross-lingual transfer to a target language includes
approaches that manipulate the vocabulary or initialize embedding for the new target vocabulary. In
vocabulary manipulation, most research has considered adding target language relevant vocabulary (Wang et al., 2020; Chau et al., 2020; Cui
et al., 2023; Larcher et al., 2023; Fujii et al., 2024;
Mundra et al., 2024; Yamaguchi et al., 2024b;
Zhao et al., 2024). Vocabulary expansion is widely
adopted for developing language-specific models
from a source model (Balachandran, 2023; Cui
et al., 2023; Fujii et al., 2024). This line of work
requires access to a large amount of target language
data. In light of this, Yamaguchi et al. (2024a) delve
into cross-lingual vocabulary expansion in lowresource settings across initialization approaches
and training strategies. Also, Zhao et al. (2024)
investigate training scales required for vocab extension and the influence of transfer on capabilities of
language generation and following instructions.
On the other hand, given that overall cost increase derived from extended vocabulary, several
cross-lingual transfer studies have pursued replacing the original vocabulary with a new target vocabulary (Zeng et al., 2023; Ostendorff and Rehm,
2023; Dobler and De Melo, 2023; Ye et al., 2024;
Liu et al., 2024; Remy et al., 2024). They focus on
initializing new target embeddings by manipulating
source embeddings according to semantic similarities between vocabularies. For instance, Gee et al.
(2022) compresses the source model’s vocabulary
using an averaging-based technique to accommodate domain-specific terms.
More recently, lines of work use source embedding and well-aligned external word embeddings
to initialize new subword embeddings for a targetspecialized vocabulary (Minixhofer et al., 2022;
Ostendorff and Rehm, 2023; Ye et al., 2024). Notably, FOCUS (Dobler and De Melo, 2023) computes a weighted mean from a multilingual language model, guided by token similarities from fastText (Bojanowski et al., 2017), to obtain new embeddings. Similarly, OFA (Liu et al., 2024) adopts
a comparable strategy to build multilingual models
and proposes a factorization–based dimension reduction embedding transfer method for efficiency.
As a hybrid method, Remy et al. (2024) combines
se
E(street)
E(road)
E(traffic)
E(sidewalk)
E(street)
E(pedestrian)
E(road)
E(traffic)
traffic street sidewalk
Unique Shared
Similarity
...
road
traffic
street
...
road
traffic
street
... ...
Linear Transfer
...
E(sidewalk)
sidewalk sidewalk
road
Figure 1: Summary of SALT. By using the paired embeddings of semantically similar tokens for each non-shared
token, we create a unique least square matrix Xti
to transfer from PLM to LLM. k denotes the number of selected
nearest tokens among overlapping tokens for linear transfer, and ht and hs refer to the hidden dimensions of the
target and source models.
the source model’s embedding under a statistical
translation scheme across languages.
However, initializing embedding relying on the
source model’s embedding inherently lacks crucial information specialized for the target language
since the source embedding does not prioritize target language adaptation. Given this concern, we
adopt embeddings from PLMs dedicated to the target language to convey richer semantic features
that the source model fails to capture during the
transfer. Based on this approach, we propose a new
cross-lingual transfer method that recycles target
language PLMs to elicit LLMs for better adaptation
in target languages.
3 SALT
We design SALT based on the assumption that
embeddings from target language-specific PLMs,
mainly pre-trained on a target language corpus,
may have richer semantic information than those
from LLMs trained in a multilingual context with
imbalanced language consideration. We explore the
way of transferring embeddings from the PLM to
the source LLM and aim to enhance generalization
on various tasks within the target language. We
show the summary of SALT framework in Figure 1
and describe the process step by step as follows:
Step 0: Objective Given source LLM (vocabulary embedding Es) with its vocabulary Vs and
target-language specific PLM (vocabulary embedding Et) with its vocabulary Vt
, we want to make
newly initialized target language embedding for Vt
to replace source LLM’s embedding while minimizing contextual misalignment and maintaining
semantically rich information inherent in Et
.
Step 1: Subword Embedding Extraction For
each Vs and Vt
, we use external static embeddings
from fastText (Bojanowski et al., 2017), which are
trained on the target languages, to extract auxiliary
embeddings. We select fastText trained in multiple languages, as it can provide embeddings of
various words using combinations of multiple subwords. For rare tokens that do not exist in fastText
or cannot be formed via subword combinations,
we initialize them from a normal distribution with
mean and standard deviation from Es.
Step 2: Estimating Similarity between Shared
and Non-shared Vocabulary As a next step, we
identify the shared vocabulary set between the Vs
and Vt
. We copy the shared vocabulary embedding
from Es. We provide further details about overlapping in Appendix A. We assume that overlapping tokens are not specific to the target language
and have likely been sufficiently learned during the
source model’s training stage, which is also done
by previous works (Dobler and De Melo, 2023; Liu
et al., 2024). Hence, our focus is on the non-shared
vocabulary set.
The most challenging task is to transfer the remaining non-shared vocabulary embeddings from
Et
into the source model’s space while preserving
semantic components. We find semantically similar tokens among shared vocabulary, Vshared =
Vs ∩ Vt
, for each non-overlapping token using extracted auxiliary embeddings. We calculate cosine
similarity scores to identify the semantically nearest token in Vshared for each vti
. We denote similarity score as in Equation 1, where f is a fastText
and vo ∈ Vshared:
sim(vti
, vo) = f(vti
) · f(vo)
∥f(vti
)∥∥f(vo)∥
(1)
Then we can get semantic similarity set Cti
for
each vti ∈/ Vshared:
Cti = {sim(vti
, vo)| ∀vo ∈ Vshared} (2)
Step 3: Building Nearest Vocabulary Set From
Cti
, we identify the top k nearest tokens. To determine the criteria of k, we use Sparsemax (Martins
and Astudillo, 2016). Sparsemax is a variant of the
softmax function that eliminates less relevant elements by assigning them values of zero, which has
been adopted in prior studies (Tran, 2020; Dobler
and De Melo, 2023).
Sparsemax on Cti
assigns weights to each paired
token’s similarity score. Most tokens assigned zero
according to the similarity distribution in Cti
are excluded. Through this process, we extract dynamic
top k nearest vocabulary subsets to build a linear
regression that transcribes non-shared tokens.
Step 4: Linear Least Square Transform Our
primary goal is to transform non-shared vocabulary
embeddings into the source model space by leveraging semantically similar shared tokens’ paired
embeddings. We employ the cheap and fast Least
Squares Transform approach. To this end, we stack
the source and target embeddings of the selected
nearest tokens, which are E′
ti
and E′
si
respectively.
Note that selected tokens have embeddings from
Es and Et since we only extract the nearest tokens
from Vshared.
We then find the transformation matrix Xti
for a
non-shared token vti
by using E′
si
as a gold label
for E′
ti
:
arg min
X∈Rht×hs
∥E
′
tiXti − E
′
si
∥ (3)
where hs and ht are the dimension of source and
target model. The solution for each Xti
can be
derived as Xti = E
′+
ti
·E′
si where E
′+
ti
is a pseudoinverse of E′
ti
(Peters and Wilkinson, 1970).
Using Xti
, we project each non-shared vocabulary embedding from Et
into Es space. As we
consider the similarity between tokens in fitting
individual least square matrix for each target token, this semantic-aware approach enables PLM’s
embedding to maintain the richness of representation in the target language during the transfer and
facilitates better alignment with the source model.
Step 5: Target Language Adaptation As a final step, we perform additional training with an
unlabeled target language corpus, called Language
Adaptive Continual Pre-training, as a final step to
align the weights between transferred embedding
and source model layers in the target language.
This stage is crucial after the initialization or transformation of the embedding since the new embedding lacks alignment with the upper layers. Thus,
this stage is considered essential as a post-transfer
task (Chau et al., 2020; Dobler and De Melo, 2023;
Liu et al., 2024; Mundra et al., 2024).