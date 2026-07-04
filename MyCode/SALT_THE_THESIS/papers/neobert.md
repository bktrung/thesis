NeoBERT: A Next-Generation BERT
Lola Le Breton
1
,
2
,
3 Quentin Fournier
2 John X. Morris
4 Mariam El Mezouar
5
Sarath Chandar
1
,
2
,
3
,
6
1Chandar Research Lab
2Mila – Quebec AI Institute
3Polytechnique Montréal
4Cornell University
5Royal Military College of Canada
6Canada CIFAR AI Chair
Abstract
Recent innovations in architecture, pre-training, and fine-tuning have led to the remarkable
in-context learning and reasoning abilities of large auto-regressive language models such as
LLaMA and DeepSeek. In contrast, encoders like BERT and RoBERTa have not seen the
same level of progress despite being foundational for many downstream NLP applications.
To bridge this gap, we introduce NeoBERT, a next-generation encoder that redefines the
capabilities of bidirectional models by integrating state-of-the-art advancements in architecture, modern data, and optimized pre-training methodologies. NeoBERT is designed for
seamless adoption: it serves as a plug-and-play replacement for existing base models, relies on an optimal depth-to-width ratio, and leverages an extended context length of 4,096
tokens. Despite its compact 250M parameter footprint, it achieves state-of-the-art results
on the massive MTEB benchmark, outperforming BERTlarge, RoBERTalarge, NomicBERT,
and ModernBERT under identical fine-tuning conditions. In addition, we rigorously evaluate
the impact of each modification on GLUE and design a uniform fine-tuning and evaluation
framework for MTEB. We release all code, data, checkpoints, and training scripts to accelerate research and real-world adoption 1,2.
1 Introduction
Auto-regressive language models have made tremendous progress since the introduction of GPT (Radford
et al., 2018), and modern large language models (LLMs) such as LLaMA 3 (Dubey et al., 2024), Mistral (Jiang
et al., 2023), OLMo (Groeneveld et al., 2024), and DeepSeek-r1 (DeepSeek-AI et al., 2025) now exhibit
remarkable reasoning and in-context learning capabilities. These improvements result from scaling both the
models and the web-scraped text datasets they are trained on, as well as from innovations in architecture
and optimization techniques. However, while decoders have continuously evolved, encoders have not seen the
same level of progress. Worse, their knowledge has become increasingly outdated despite remaining critical
for a wide range of downstream NLP tasks that depend on their embeddings, notably for retrieval-augmented
generation (Ram et al., 2023) and toxicity classification (Hartvigsen et al., 2022). Despite being five years
old, BERT (Devlin et al., 2019) and RoBERTa (Liu et al., 2019) remain widely used, with more than 110
million combined downloads from Hugging Face as of the writing of this paper.
Similar to decoders, which undergo multi-stage processes of pre-training, instruction-tuning, and alignment,
encoders also require successive training phases to perform well on downstream tasks. First, encoders go
through self-supervised pre-training on large corpora of text with the masked language modeling objective.
By predicting masked or replaced tokens, this stage enables models to learn the structural patterns of
language and the semantics of words. However, the pre-training task is disconnected from downstream
applications, and models require additional training to achieve strong performance in clustering or retrieval.
Thus, a second fine-tuning phase is often achieved through multiple stages of semi-supervised contrastive
learning, where models learn to differentiate between positive and negative sentence pairs, refining their
embeddings in the latent space. 1https://huggingface.co/chandar-lab/NeoBERT 2https://github.com/chandar-lab/NeoBERT
1
Recently, substantial progress has been made in improving the fine-tuning stage of pre-trained encoders,
with models like GTE (Li et al., 2023b), jina-embeddings (Sturua et al., 2024), SFR-embeddings (Meng
et al., 2024), and CDE (Morris & Rush, 2024) significantly outperforming previous encoders on the MTEB
leaderboard, a recent and challenging benchmark spanning 7 tasks and 56 datasets. However, all these
approaches focus on proposing complex fine-tuning methods and do not address the inherent limitations of
their pre-trained backbone models.
As a result, there is a dire need for a new generation of BERT-like pre-trained models that incorporate
up-to-date knowledge and leverage both architectural and training innovations, forming stronger backbones
for these more advanced fine-tuning procedures.
In response, we introduce NeoBERT, a next-generation BERT that integrates the latest advancements in
architecture, data, and pre-training strategies. The improvements are rigorously validated on GLUE by fully
pre-training 10 different models that successively incorporate the modifications. This validation ensures that
the improvements benefit encoder architectures and highlights how some design choices drastically affect
the model’s abilities. Additionally, we design and experimentally validate a two-stage training procedure to
increase NeoBERT’s maximum context window from 1, 024 to 4, 096. To ensure a fair evaluation of NeoBERT
against existing baselines and to isolate the impact of fine-tuning procedures, we propose a model-agnostic
and systematic fine-tuning strategy with straightforward contrastive learning. All models are fine-tuned
using this standardized approach and subsequently evaluated on the MTEB benchmark.
On MTEB, NeoBERT consistently outperforms all competing pre-trained models while being 100M parameters smaller than the typical large-sized encoders. With a context window of 4,096 tokens, it processes
sequences 8× longer than RoBERTa (Liu et al., 2019) and two times longer than NomicBERT (Nussbaum
et al., 2024). It is also the fastest encoder of its kind, significantly outperforming ModernBERT base and
large in terms of inference speed. Despite its compact 250M parameter size, NeoBERT is trained for over 2T
tokens, prioritizing training over scale to maximize accessibility for both academic researchers and industry
users without requiring large-scale compute resources. This makes NeoBERT the most extensively trained
model among modern encoders, ensuring robust generalization and superior downstream performance. Furthermore, NeoBERT maintains the same hidden size as base models, allowing for seamless plug-and-play
adoption without modifications to existing architectures. As the only fully open-source model of its kind, we
release the code, data, training scripts, and model checkpoints, reinforcing our commitment to reproducible
research.
2 Related work
In 2019, Devlin et al. (2019) introduced BERT, a novel approach to embedding text using bi-directional
Transformers pre-trained without supervision on large corpora. Shortly after, Liu et al. (2019) improved
over BERT’s pre-training by removing the next-sentence prediction objective and drastically increasing the
amount of data, leading to RoBERTa. Since then, the primary focus of the community has shifted towards
optimizing the fine-tuning phase of these models through contrastive learning, where the model is trained to
maximize the similarity between positive text pairs while pushing them apart from negative samples.
Among the earliest contrastive learning approaches designed for encoders, SimCSE (Gao et al., 2022) demonstrated that sentence pairs could be easily generated by feeding the same input to the model twice and
applying dropout to introduce noise. However, this simple approach was soon outperformed by models like
GTE (Li et al., 2023b), which introduced more advanced contrastive learning techniques. GTE employed
a weakly supervised stage that takes advantage of the vast number of successive sentence pairs available
in traditional unlabeled datasets, followed by a semi-supervised stage incorporating labeled sentence pairs
from high-quality datasets such as NLI (Bowman et al., 2015) and FEVER (Thorne et al., 2018). Recently,
fine-grained strategies have emerged to better adapt models to both task and context. For instance, Jinaembeddings (Sturua et al., 2024) introduced task-specific Low-Rank Adaptation (LoRA) adapters. As of
January 2025, CDE (Morris & Rush, 2024) ranks at the top of the MTEB leaderboard for models under
250M parameters thanks to two key innovations: grouping samples with related contexts into the same batch
and providing contextual embeddings for the entire corpus in response to individual queries.
2
However, pre-training has not seen the same level of effort, and thus progress, most likely due to its prohibitively high computational cost. RoBERTa, for instance, required a total of 1, 024 V100 days for its pretraining. As a result, GTE, Jina-embeddings, and CDE all rely on pre-trained BERT, XLM-RoBERTa (Conneau et al., 2020), and NomicBERT (Nussbaum et al., 2024) to initialize their respective models. The latter,
NomicBERT, represents a recent effort to refine BERT’s architecture and pre-training. NomicBERT incorporates architectural improvements such as SwiGLU and RoPE, utilizes FlashAttention, and extends the
context length to 2, 048 tokens. Despite these innovations, NomicBERT still relied on sub-optimal choices,
as discussed in section 3. In parallel with the development of NeoBERT, Warner et al. (2024) released
ModernBERT with the goal of further refining the pre-training of NomicBERT. Although we share some of
the modifications, we make distinct design choices and conduct thorough ablations that ultimately lead to
greater performance on MTEB.
3 NeoBERT
The following section describes NeoBERT’s improvements over BERT and RoBERTa, as well as the recent
NomicBERT and ModernBERT models. Since GTE and CDE use BERT and NomicBERT as their pretrained backbone, they inherit their respective characteristics. Table 1 summarizes the modifications.
Table 1: Comparison of Model Architectures, Training Data, and Pre-Training Configurations.
BERT RoBERTa NomicBERT ModernBERT NeoBERT
base large base large base base large medium
Layers 12 24 12 24 12 22 28 28
Hidden Size 768 1, 024 768 1, 024 768 768 1, 024 768
Attention Heads 12 16 12 16 12 12 16 12
Parameters 120M 350M 125M 355M 137M 149M 395M 250M
Activation Function GeLU SwiGLU GeGLU SwiGLU
Positional Encoding Positional Embeddings RoPE RoPE RoPE
Normalization Post-LayerNorm Post-LayerNorm Pre-LayerNorm Pre-RMSNorm
Data Sources BooksCorpus
Wikipedia
BooksCorpus
OpenWebText
Stories / CC-News
BooksCorpus
Wikipedia
Undisclosed RefinedWeb
Dataset Size 13GB 160GB 13GB - 2.8TB
Dataset Year 2019 2019 2023 - 2023
Tokenizer Level Character Byte Character Character Character
Vocabulary Size 30K 50K 30K 50K 30K
Sequence Length 512 512 2, 048 1, 024 → 8, 192 1, 024 → 4, 096
Objective MLM + NSP MLM MLM MLM MLM
Masking Rate 15% 15% 30% 30% 20%
Masking Scheme 80/10/10 80/10/10 - - 100
Optimizer Adam Adam AdamW StableAdamW AdamW
Scheduler - - - WSD CosineDecay
Batch Size 131k tokens 131k 8M 448k to 5M 2M
Tokens Seen 131B 131B - ∼ 2T 2.1T
Training DDP DDP
DeepSpeed
FlashAttention
Alternate Attention
Unpadding
FlashAttention
DeepSpeed
FlashAttention
3
3.1 Architecture
The Transformer architecture has been refined over the years and has now largely stabilized, with models like
LLaMA 3 being essentially the same as the original LLaMA. NeoBERT integrates the latest modifications
that have, for the most part, become standard.
Depth-to-Width The concept of depth efficiency has long been recognized in neural network architectures.
In the case of Transformers, stacking self-attention layers is so effective that it can quickly saturate the
network’s capacity. Recognizing this, Levine et al. (2020) provided theoretical and empirical evidence for
an optimal depth-to-width ratio in Transformers. Their findings suggested that most language models
were operating in a “depth-inefficiency” regime, where allocating more parameters to width rather than
depth would have improved performance. In contrast, small language models like BERT, RoBERTa, and
NomicBERT are instead in a width-inefficiency regime. To maximize NeoBERT’s parameter efficiency while
ensuring it remains a seamless plug-and-play replacement, we retain the original BERTbase width of 768 and
instead increase its depth to achieve this optimal ratio.
Positional Information Transformers inherently lack the ability to distinguish token positions. Early
models like BERT and RoBERTa addressed this by adding absolute positional embeddings to the token
embeddings before the first Transformer block. However, this approach struggles to generalize to longer
sequences and requires the positional information to be propagated across layers. To overcome these limitations, Su et al. (2023) proposed Rotary Position Embeddings (RoPE), which integrate relative positional
information directly into the self-attention mechanism. RoPE has quickly become the default in modern
Transformers due to its significant improvements in performance and extrapolation capabilities. NeoBERT,
like all newer encoders, integrates RoPE. Nevertheless, degradation still occurs with sequences significantly
longer than those seen during training. As a solution, Peng et al. (2023) introduced Yet Another RoPE
Extension (YaRN), which allows to efficiently fine-tune models on longer contexts. NeoBERT is readily
compatible with YaRN, making it well-suited for tasks requiring extended context.
Layer Normalization Consistent with standard practices in modern Transformer architectures, we move
the normalization layer inside the residual connections of each feed-forward and attention block, a technique
known as Pre-Layer Normalization (Pre-LN). Pre-LN improves stability, allows for larger learning rates,
and accelerates model convergence (Xiong et al., 2020). While all newer encoder models adopt Pre-LN,
they typically continue to use the classical LayerNorm rather than Root Mean Square Layer Normalization
(RMSNorm). In NeoBERT, we substitute the classical LayerNorm with RMSNorm (Zhang & Sennrich,
2019), which achieves comparable training stability while being slightly less computationally intensive, as it
requires one fewer statistic.
Activations BERT and RoBERTa utilize the standard Gaussian Error Linear Unit (GELU) activation
function. However, Shazeer (2020) demonstrated the benefits of the Gated Linear Unit in Transformer
architectures. These activation functions have since been adopted in several language models, including
the LLaMA family. Following previous works, NeoBERT incorporates the SwiGLU activation function, and
because it introduces a third weight matrix, we scale the number of hidden units by a factor of 2
3
to keep
the number of parameters constant.
3.2 Data
Data has emerged as one of the most critical aspects of pre-training, and datasets with increasing quantity
and diversity are frequently released. NeoBERT takes advantage of the latest datasets that have proven to
be effective.
Dataset BERT and NomicBERT were pre-trained on two carefully curated and high-quality datasets:
Wikipedia and BookCorpus (Zhu et al., 2015). As Baevski et al. (2019) demonstrated that increasing data
size can improve downstream performance, Liu et al. (2019) pre-trained RoBERTa on 10 times more data
from BookCorpus, CC-News, OpenWebText, and Stories. However, RoBERTa’s pre-training corpus has
4
become small in comparison to modern web-scraped datasets built by filtering and deduplicating Common
Crawl dumps. Following the same trend, we pre-trained NeoBERT on RefinedWeb (Penedo et al., 2023),
a massive dataset containing 600B tokens, nearly 18 times larger than RoBERTa’s. Although RefinedWeb
does not have strict high-quality constraints, we believe that exposing the model to such a large and diverse
dataset will improve its real-world utility.
Sequence Length BERT and RoBERTa were pre-trained on sequences up to 512 tokens, which limits
their downstream utility, especially without RoPE and YaRN. NomicBERT increased the maximum length
to 2, 048 and employed Dynamic NTK interpolation at inference to scale to 8192. To further broaden
NeoBERT’s utility, we seek to increase the context length. However, due to the computational cost associated
with pre-training, we adopt a two-stage pre-training procedure similar to LLMs like LLaMA 3. In the first
stage, we train the model for 1M steps (2T tokens) using sequences truncated to a maximum length of
1, 024 tokens, referring to this version as NeoBERT1024. In the second stage, we extend the training for an
additional 50k steps (100B tokens), increasing the maximum sequence length to 4, 096 tokens. We refer to
this final model as NeoBERT4096. To ensure the model encounters longer sequences during this stage, we
create two additional sub-datasets, Refinedweb1024+ and Refinedweb2048+, containing only sequence lengths
greater than 1, 024 and 2, 048 tokens, respectively, alongside the original Refinedweb dataset. Each batch
is sampled from Refinedweb, Refinedweb1024+ and Refinedweb2048+ with probabilities 20%, 40%, and 40%.
Since longer sequences tend to represent more complex or academic content, this strategy helps mitigate
the distribution shift typically observed when filtering for longer sequences. We explore the benefits of this
two-stage training strategy in subsection 5.3.
3.3 Pre-Training
Encoder pre-training has received less attention than the data and architecture. However, many improvements made to decoders also apply to encoders. NeoBERT combines encoder-specific modifications with
widely accepted decoder improvements.
Objective In light of RoBERTa’s findings that dropping the next-sentence prediction task does not harm
performance, both NomicBERT and NeoBERT were only pre-trained on masked language modeling. Moreover, Wettig et al. (2023) challenged the assumption that the 15% masking rate of BERT and RoBERTa is
universally optimal. Instead, their findings suggest that the optimal masking rate is actually 20% for base
models and 40% for large models. Intuitively, a model learns best when the difficulty of its training tasks
aligns with its capabilities. Based on their insight, we increase the masking rate to 20%, while NomicBERT
exceeds it by opting for 30%.
Optimization Following standard practice, we use the AdamW optimizer (Loshchilov & Hutter, 2019) with
the same hyperparameters as LLaMA 2: β1 = 0.9, β2 = 0.95, and ϵ = 10−8
. In preliminary experiments,
we also considered SOAP (Vyas et al., 2025), a recent extension of the Shampoo optimizer, but it failed to
outperform Adam and AdamW and has been omitted from the list of ablations. We employ a linear warmup
for 2, 000 steps to reach a peak learning rate of 6 × 10−4
, followed by a cosine decay to 10% of the peak
learning rate over 90% of the training steps. Once fully decayed, the learning rate remains constant for the
last 100k steps at a sequence length of 1, 024 and 50k steps at a sequence length of 4, 096. We use a weight
decay of 0.1 and apply gradient clipping with a maximum norm of 1.0.
Scale Recent generative models like the LLaMA family (Touvron et al., 2023; Dubey et al., 2024) have
demonstrated that language models benefit from being trained on significantly more tokens than was previously standard. Recently, LLaMA-3.2 1B was successfully trained on up to 9T tokens without showing signs
of saturation. Moreover, encoders are less sample-efficient than decoders since they only make predictions
for masked tokens. Therefore, it is reasonable to believe that encoders of similar sizes can be trained on an
equal or even greater number of tokens without saturating. For NeoBERT’s pre-training, we use batch sizes
of 2M tokens over 1M steps in the first stage and 50k steps in the second, resulting in a theoretical total of
2.1T tokens. Note that because sequences are padded to the maximum length, this represents a theoretical
number of tokens. In terms of tokens, this is comparable to RoBERTa and represents a 2x increase over
5
Table 2: Modifications between successive ablations. The initial M0 baseline corresponds to a model similar
to BERT, while M9 corresponds to NeoBERT.
Modification Before After
M1
Embedding Positional RoPE
Activation GELU SwiGLU
Pre-LN LayerNorm RMSNorm
M2 Dataset Wiki + Book RefinedWeb
M3 Tokenizer Google WordPiece LLaMA BPE
M4
Optimizer Adam AdamW
Scheduler Linear Cosine
M5 Masking Scheme 15% (80 / 10 / 10) 20% (100)
M6 Sequence packing False True
M7 Model Size 120M 250M
M8 Depth - Width 16 - 1056 28 - 768
M9
Batch size 131k 2M
Context length 512 4, 096
NomicBERT. In terms of training steps, this amounts to a 2x increase over RoBERTa and a 10x increase
over NomicBERT.
Efficiency We improve efficiency by parallelizing the model across devices using DeepSpeed (Aminabadi
et al., 2022) with the ZeRO (Rajbhandari et al., 2020) optimizer, reducing memory usage by eliminating
data duplication across GPUs and increasing the batch size. We further optimize the GPU utilization by
employing fused operators from the xFormers library to reduce overhead, selecting all dimensions to be
multiples of 64 to align with GPU architectures, and removing biases to simplify computations without
sacrificing performance. To address the quadratic demands of attention, we integrate FlashAttention (Dao,
2023), which computes exact attention without storing the full matrices.
4 Effect of Design Choices
We conduct a series of ablations in controlled settings to evaluate our improvements to the original BERT
architecture. We fully train each model for 1M steps, controlling for the seed and dataloader states to
ensure successive models are trained with identical setups. These resource-intensive ablations were crucial to
confirm our design choices, as they are based on the literature of pre-training decoder models. The baseline
model, referred to as M0, is similar to BERTbase but includes pre-layer normalization. Following RoBERTa,
M0 also drops the next sentence prediction objective. We introduce modifications iteratively, resulting in
a total of ten different models, as detailed in Table 2. To mitigate computational costs, the ablations are
evaluated on the GLUE benchmark with a limited hyperparameter grid search of batch sizes ∈ {16, 32} and
learning rates ∈ {1e − 5, 2e − 5, 3e − 5}. For the final model M10, we extend the grid search, as detailed in
Appendix C. Results are in Figure 1.
Key Performance-Enhancing Modifications As expected, the two modifications that had the greatest
impact on the average GLUE score were related to scale. In M2, replacing Wikitext and BookCorpus with
the significantly larger and more diverse RefinedWeb dataset improved the score by +3.6%, while increasing
the model size from 120M to 250M in M7 led to a +2.9% relative improvement. Note that to assess the
impact of the depth-to-width ratio, we first scale the number of parameters in M7 to 250M while maintaining
6
Figure 1: GLUE ablation scores on the development set. All modifications are cumulative, except for M3
and M6 in grey, which are not included in the subsequent models. Increasing data size and diversity leads
to the highest relative improvement (M2, +3.6%), followed by the model size (M7, +2.9%). Packing the
sequences and using the LLaMA 2 tokenizer cause the largest relative drops (M6, −2.9%, M3, −2.1%).
0 10 20 30 40 50 60 70 80 90
Baseline M0
RoPE, SwiGLU, RMSNorm M1
RefinedWeb M2
LLaMA Tokenizer M3
AdamW, Cosine Decay M4
20% Masking M5
Sequence Packing M6
250M Parameters M7
Optimal Depth-to-Width M8
Training Scale M9
Larger Grid Search M10
79.2
81.1
84.0
82.2
83.5
82.9
80.5
85.3
86.0
87.3
89.0
a similar ratio to BERTbase, resulting in 16 layers of dimension 1056. In M8, the ratio is then adjusted to
28 layers of dimension 768.
Modifications That Were Discarded In M3, replacing the Google WordPiece tokenizer with LLaMA
BPE results in a −2.1% performance decrease. The performance gap observed likely stems from inherent
differences between the algorithms, training corpus, and design strategies employed by the two tokenizers.
We further discuss this in Appendix B. In M6, we unpad the sequences by concatenating samples of the
same batch. While this removes unnecessary computation on padding tokens, packing sequences without
accounting for cross-sequence attention results in a relative performance drop of −2.8%. Although this
modification was discarded from our subsequent ablations, we incorporate methods of un-padding with
accurate cross-attention in our released codebase, following Kundu et al. (2024).
Modifications Retained Despite Performance Trade-offs Unexpectedly, using AdamW (Loshchilov
& Hutter, 2019) and cosine decay (Loshchilov & Hutter, 2017) in M4 decreases performance by −0.5%.
As AdamW introduces additional regularization with weight decay, we expect that it will become beneficial
when extending training by mitigating overfitting. Similarly, increasing the masking ratio from 15% to 20%
in M5 decreases performance by −0.7%. We hypothesize that increasing the task difficulty initially hinders
downstream task performance but is likely to become advantageous when training larger models on more
tokens. Consequently, we retain both modifications despite being unable to verify these hypotheses at scale
due to the computational costs.
5 Experiments
Selecting appropriate metrics and benchmarks is crucial for properly assessing the downstream performance
and utility of language models. Following both early and recent studies, we include the GLUE and MTEB
benchmarks in our evaluations.
7
5.1 GLUE
The GLUE benchmark (Wang et al., 2019) is a cornerstone of language modeling evaluations and has played
a significant role in the field. Although GLUE is now 6 years old and the community has long recognized its
limitations, we report the GLUE score due to its widespread adoption and to facilitate the comparison of
NeoBERT with existing encoders. Following standard practices, we fine-tune NeoBERT on the development
set of GLUE with a classical hyperparameter search and introduce transfer learning between related tasks.
Complete details of the fine-tuning and best hyperparameters are presented in Appendix C. NeoBERT
achieves a score of 89.0% comparable to previous large models while being 100M to 150M parameters
smaller. We present the results in Table 3.
Table 3: GLUE scores on the development set. Baseline scores were retrieved as follows: BERT from Table
1 of Devlin et al. (2019), RoBERTa from Table 8 of Liu et al. (2019), DeBERTa from Table 3 of He et al.
(2023), NomicBERT from Table 2 of Nussbaum et al. (2024), GTE from Table 13 of Zhang et al. (2024),
and ModernBERT from Table 5 of Warner et al. (2024).
Size Model MNLI QNLI QQP RTE SST MRPC CoLA STS Avg.
Base
(≤ 150M)
BERT 84.0 90.5 71.2 66.4 93.5 88.9 52.1 85.8 79.6
RoBERTa 87.6 92.8 91.9 78.7 94.8 90.2 63.6 91.2 86.4
GTE-en-8192 86.7 91.9 88.8 84.8 93.3 92.1 57.0 90.2 85.6
NomicBERT2048 86.0 92.0 92.0 82.0 93.0 88.0 50.0 90.0 84.0
ModernBERT 89.1 93.9 92.1 87.4 96.0 92.2 65.1 91.8 88.5
Medium
250M
NeoBERT1024 88.9 93.9 90.7 91.0 95.8 93.4 64.8 92.1 88.8
NeoBERT4096 89.0 93.7 90.7 91.3 95.6 93.4 66.2 91.8 89.0
Large
(≥ 340M)
BERT 86.3 92.7 72.1 70.1 94.9 89.3 60.5 86.5 82.1
RoBERTa 90.2 94.7 92.2 86.6 96.4 90.9 68.0 92.4 88.9
DeBERTaV3 91.9 96.0 93.0 92.7 96.9 91.9 75.3 93.0 91.4
GTE-en-8192 89.2 93.9 89.2 88.1 95.1 93.5 60.4 91.4 87.6
ModernBERT 90.8 95.2 92.7 92.1 97.1 91.7 71.4 92.8 90.5
5.2 MTEB
Beyond the GLUE benchmark, we consider the more recent and challenging MTEB benchmark (Muennighoff
et al., 2023), which has quickly become a standard for evaluating embedding models, with a wide coverage
of 7 tasks and 56 datasets in its English subset.
MTEB tasks rely on the cosine similarity of embeddings pooled across tokens in a sentence. The most common and straightforward pooling strategy is computing the average of each token’s final hidden representation. However, because out-of-the-box encoders are trained with the masked language modeling objective,
they provide no guarantee that mean embeddings will produce meaningful representations without further
fine-tuning. As a result, models require expensive fine-tuning strategies crafted for MTEB to achieve strong
scores. For instance, GTE (Li et al., 2023b) with multi-stage contrastive learning, Jina-embeddings (Sturua
et al., 2024) with task-specific Low-Rank Adaptation (LoRA) adapters, and CDE (Morris & Rush, 2024),
with batch clustering and contextual corpus embeddings all pushed the limits of the leaderboard in their
respective categories.
These coupled stages make it challenging to disentangle the respective impacts of pre-training and fine-tuning
on the final model’s performance. To isolate and fairly evaluate the improvements introduced during pretraining, we implemented an affordable, model-agnostic fine-tuning strategy based on classical contrastive
learning. This fine-tuning approach was designed in accordance with established methods in the literature.
Its controlled settings ensured that all models were fine-tuned and evaluated under identical conditions.
8
5.2.1 Unified Contrastive Learning
Method Given a dataset of positive pairs D = {qi
, d+
i
}
n
i=1, a similarity metric s, a temperature parameter
τ , and a set of negative documents Nq for each query q, the contrastive loss is defined as:
L = − log e
s(q,d+)/τ
e
s(q,d+)/τ +
P
d−∈Nq
e
s(q,d−)/τ
Negative documents can be either generic samples of the same format or tailored hard negatives, which
exhibit a high degree of similarity to the contrasted sample in their original representation. We constructed
a dataset of positive query-document pairs with optional hard negatives based on open-source datasets
previously employed in the literature (Li et al., 2023a) for a total of nine million documents. In addition
to the optional hard negatives, we also leverage in-batch, task-homogeneous negatives. In line with prior
research (Li et al., 2023a), we employ task-specific instructions and temperature-scaled sampling of the
datasets. Complete details about the data, training, and evaluation can be found in Appendix D.
Results We found that training for more than 2,000 steps resulted in minimal performance gains. Table 4
presents the complete MTEB-English evaluation of all fine-tuned models. Although NeoBERT is 100M
parameters smaller than all large baselines, it is the best model overall with a +4.5% relative increase over
the second best model, demonstrating the benefits of its architecture, data, and pre-training improvements.
Table 4: MTEB(eng, v1) scores after 2,000 steps of fine-tuning with our unified contrastive learning.
Size Model Class. Clust. PairClass. Rerank. Retriev. STS Summ. Avg. 12 tasks 11 tasks 3 tasks 4 tasks 15 tasks 10 tasks 1 tasks
Base
BERT 60.6 37.0 71.5 48.9 28.3 69.9 31.1 48.1
RoBERTa 58.7 36.7 71.2 49.8 26.9 71.8 29.1 47.7
DeBERTaV3 45.9 15.2 44.3 39.0 3.5 42.2 25.0 26.9
NomicBERT2048 55.0 35.3 69.0 48.8 30.5 70.1 30.1 47.1
ModernBERT 58.9 38.1 63.8 48.5 21.0 66.2 30.1 45.0
Medium NeoBERT4096 61.6 40.8 76.2 51.2 31.6 74.8 30.7 51.3
Large
BERT 59.8 39.3 70.9 49.7 29.6 71.4 31.2 49.1
RoBERTa 57.1 39.3 72.5 51.3 30.0 71.7 31.1 48.9
DeBERTaV3 45.9 16.4 45.0 40.8 4.0 40.1 29.9 27.1
ModernBERT 62.4 38.7 65.5 50.1 23.1 68.3 27.8 46.9
5.2.2 Contextual Document Embeddings
While a unified fine-tuning framework enables fair model comparisons, computationally intensive fine-tuning
techniques are necessary to achieve top-ranking MTEB scores. We apply one such strategy, Contextual
Document Embeddings (CDE) (Morris & Rush, 2024), to NeoBERT. CDE previously led the MTEB leaderboard with NomicBERT (cde-v1) and ModernBERT (cde-v2) for English models under 400M parameters.
NeoBERT outperforms both, achieving an average score of 66.60, the highest in its parameter class. See
Appendix G for more details on the fine-tuning.
Table 5: MTEB(eng, v1) scores after fine-tuning with CDE. ∗Ranks on MTEB(eng, v1) for models under
400M parameters as of April 2025.
Model Class. Clust. PairClass. Rerank. Retriev. STS Summ. Avg. Rank∗
12 tasks 11 tasks 3 tasks 4 tasks 15 tasks 10 tasks 1 tasks
NomicBERT 81.72 48.32 84.69 56.75 53.27 81.64 31.23 65.00 3
ModernBERT 80.62 49.48 85.23 56.94 54.19 83.30 31.60 65.68 2
NeoBERT 82.14 50.29 86.71 57.71 56.37 82.30 30.98 66.60 1
9
5.3 Sequence Length
Following previous literature, NeoBERT underwent an additional 50k pre-training steps, during which it
was exposed to extended sequences of up to 4,096 tokens. To assess the impact of this additional training,
we randomly sampled 2,467 long sequences from the English subset of Wikipedia. For each sequence, we
independently masked each token at position i and computed its cross-entropy loss, li
. The pseudo-perplexity
of the entire sentence is then defined as P = exp
1
n
Pn
i=1 li

. We present the results in Figure 2.
Figure 2: Pseudo-Perplexity in function of the sequence length for NeoBERT1024 (left) and NeoBERT4096
(right). This validates the effectiveness of the final pre-training stage on NeoBERT’s ability to model long
sequences.
1000 1500 2000 2500 3000 3500 4000
Sequence Length
0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
Pseudo-Perplexity
NeoBERT1024
1000 1500 2000 2500 3000 3500 4000
Sequence Length
0.0
2.5
5.0
7.5
10.0
12.5
15.0
17.5
20.0
Pseudo-Perplexity
NeoBERT4096
Although NeoBERT1024 was trained exclusively on sequences of up to 1, 024 tokens, it generalizes effectively
to context lengths approaching 3,000 tokens. This demonstrates the robustness of RoPE embeddings to outof-distribution inputs. Moreover, after an additional 50k training steps with sequences up to 4, 096 tokens,
NeoBERT4096 successfully models longer sequences. This approach provides a compute-efficient strategy for
extending the model’s maximum context window beyond its original length. To further test NeoBERT’s
generalization capabilities, we extend the context beyond what was seen during training and present results
in Appendix E.
5.4 Efficiency
To assess model efficiency, we construct a synthetic dataset consisting of maximum-length sequences of sizes
{512, 1024, 2048, 4096, 8192}. For each sequence length, we scale the batch size from 1 to 512 samples or
until encountering out-of-memory errors. Inference is performed for 100 steps on a single A100 GPU, and we
report the highest throughput achieved for each model and sequence length. Figure 3 presents the results.
Due to their low parameter count and relatively simple architecture, BERT and RoBERTa are the most efficient for sequences up to 512 tokens. However, their use of positional embeddings prevents them from further
scaling the context window. For extended sequences, NeoBERT significantly outperforms ModernBERTbase,
despite having 100M more parameters, achieving a 46.7% speedup on sequences of 4, 096 tokens.
6 Discussion
Encoders are compact yet powerful tools for language understanding and representation tasks. They require
fewer parameters and significantly lower training costs compared to their decoder counterparts, making them
strong alternatives for applications that do not involve text generation. Traditionally, the representational
capacity of these models has been assessed through downstream tasks such as classification, in particular
through the GLUE benchmark.
10
Figure 3: Model throughput (tokens per second) as a function of sequence length (↑ is better). Above 1, 024
in sequence length, NeoBERT surpasses ModernBERTbase despite having 100M more parameters.
512 1024 2048 4096 8192
5k
10k
15k
20k
25k
Throughpout (tokens/second)
ModernBERTbase
BERTbase
BERTlarge
ModernBERTlarge
NeoBERT
RoBERTalarge
RoBERTabase
While GLUE has played a pivotal role in guiding model adoption, it includes only nine sequence classification
datasets, four of which are entailment tasks. Moreover, its small dataset sizes and occasionally ambiguous
labeling make it prone to overfitting, with models long surpassing human performance on the benchmark.
Although DeBERTa-v3 achieves state-of-the-art performance on GLUE by a significant margin, our finetuning experiments reveal its comparatively poor performance on the more recent MTEB benchmark. MTEB
encompasses a broader range of datasets and tasks, but attaining high performance on its leaderboard
necessitates carefully crafted fine-tuning strategies with costly training requirements. As more complex
fine-tuning strategies emerge, it becomes unclear what the source of score improvements is. Moreover,
these strategies are not easily reproducible or accessible, limiting the possibility of fair comparison between
pre-trained backbones.
This underscores the limitations of current evaluation paradigms and highlights the need for more affordable
and standardized frameworks. We advocate for future research to focus on the development of standardized fine-tuning protocols and the exploration of new zero-shot evaluation methodologies to ensure a more
comprehensive and unbiased assessment of encoder-only models.
7 Conclusion
We introduced NeoBERT, a state-of-the-art encoder pre-trained from scratch with the latest advancements
in language modeling, architecture, and data selection. To ensure rigorous validation, we systematically
evaluated every design choice by fully training and benchmarking ten distinct models in controlled settings.
On GLUE, NeoBERT outperforms BERTlarge and NomicBERT and is comparable with RoBERTalarge
despite being 100M parameters smaller and supporting sequences eight times longer. To further validate
its effectiveness, we conducted a comprehensive evaluation on MTEB, carefully isolating the effects of pretraining and fine-tuning to provide a fair comparison against the best open-source embedding models. Under
identical fine-tuning conditions, NeoBERT consistently outperforms all baselines. In addition, when finetuned with more advanced techniques, NeoBERT ranks at the top of the MTEB leaderboard for models under
400M parameters. With its unparalleled efficiency, optimal depth-to-width, and plug-and-play compatibility,
NeoBERT represents the next generation of encoder models. To foster transparency and collaboration, we
release all code, data, model checkpoints, and training scripts, making NeoBERT the only fully open-source
model of its kind.
11
Broader Impact Statement
Despite its improvements, NeoBERT inherits the biases and limitations of its pre-training data. Moreover,
the greatest jump in performance stems from the pre-training dataset, and as newer, larger, and more diverse
datasets become available, retraining will likely be needed to further improve its performance. Nonetheless,
NeoBERT stands today as an affordable state-of-the-art pre-trained encoder with great potential for downstream applications.
Acknowledgements
Sarath Chandar is supported by the Canada CIFAR AI Chairs program, the Canada Research Chair in
Lifelong Machine Learning, and the NSERC Discovery Grant. Quentin Fournier is supported by the Lambda
research grant program. The authors acknowledge the computational resources provided by Mila and the
Royal Military College of Canada.