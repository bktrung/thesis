ViDeBERTa: A powerful pre-trained language model for Vietnamese
Cong Dao Tran ∗
FPT Software AI Center
daotc2@fsoft.com.vn
Nhut Huy Pham ∗
FPT Software AI Center
huypn10@fsoft.com.vn
Anh Nguyen
Microsoft
anhnguyen@microsoft.com
Truong Son Hy †
University of California San Diego
tshy@ucsd.edu
Tu Vu
University of Massachusetts Amherst
tuvu@cs.umass.edu
Abstract
This paper presents ViDeBERTa, a new
pre-trained monolingual language model
for Vietnamese, with three versions -
ViDeBERTaxsmall, ViDeBERTabase, and
ViDeBERTalarge, which are pre-trained on a
large-scale corpus of high-quality and diverse
Vietnamese texts using DeBERTa architecture. Although many successful pre-trained
language models based on Transformer have
been widely proposed for the English language,
there are still few pre-trained models for Vietnamese, a low-resource language, that perform
good results on downstream tasks, especially
Question answering. We fine-tune and evaluate
our model on three important natural language
downstream tasks, Part-of-speech tagging,
Named-entity recognition, and Question
answering. The empirical results demonstrate
that ViDeBERTa with far fewer parameters
surpasses the previous state-of-the-art models
on multiple Vietnamese-specific natural
language understanding tasks. Notably,
ViDeBERTabase with 86M parameters, which
is only about 23% of PhoBERTlarge with
370M parameters, still performs the same or
better results than the previous state-of-the-art
model. Our ViDeBERTa models are available
at: https://github.com/HySonLab/ViDeBERTa.
1 Introduction
In recent years, pre-trained language models
(PLMs) and Transformer-based architecture models have been essential in the advancement of
Natural Language Processing (NLP). Large-scale
Transformer-based pre-trained models with the capacity to derive a contextual representation of the
languages in the training data include GPT (Radford et al., 2019; Brown et al., 2020), BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019), XLNet (Yang et al., 2019b), ELECTRA (Clark et al.,
2020), T5 (Raffel et al., 2020), and DeBERTa (He
∗
: Co-first authors. †: Correspondent author.
et al., 2020, 2021). Following pre-training, these
models performed at the cutting edge on various
downstream NLP tasks (Devlin et al., 2019). The
development of pre-trained models in other languages, including Vietnamese (PhoBERT (Nguyen
and Nguyen, 2020); ViBERT (Tran et al., 2020);
ViT5 (Phan et al., 2022)), and Arabic (Antoun et al.,
2021), has been spurred on by the success of pretrained models in English. In order to enhance performance across several languages by learning both
general and language-specific representations, multilingual pre-trained models ( XLM-R (Conneau
et al., 2020), mT5 (Xue et al., 2021), and mBART
(Liu et al., 2020) are also being developed.
Most recently, PhoBERT (Nguyen and Nguyen,
2020), the first large pre-trained model for Vietnamese that inherits the RoBERTa (Liu et al.,
2019) architecture, has demonstrated the effectiveness of the trained language model compared with
current methods modernized in four Vietnamesespecific tasks, including Part of Speech Tagging
(POS), Dependency Parsing, Named Entity Recognition (NER), and Natural Language Inference
(NLI). Nevertheless, there are still rooms to build
an improved pre-trained language model for Vietnamese. Firstly, PhoBERT was pre-trained on a
relatively small Vietnamese dataset of 20GB of uncompressed texts, while pre-trained language models can be significantly improved by using more
pre-training data (Liu et al., 2019). Secondly, Question answering (QA) is one of the most impactful
tasks that has mainly focused on the computational
linguistics and artificial intelligence research community within information retrieval and information extraction in recent years. However, there are
a few pre-trained models for Vietnamese that produce efficient results in the QA tasks, especially
PhoBERT (Nguyen and Nguyen, 2020) and ViT5
(Phan et al., 2022). Last but not least, some previous works point to DeBERTa architecture (He
et al., 2020, 2021) using several novel techniques
1071
that can significantly outperform RoBERTa and improve the efficiency of model pre-training and the
performance of downstream tasks in some respects.
Inspired by that, we introduce an improved largescale pre-trained language model, ViDeBERTa,
trained on CC100 Vietnamese monolingual, following the architecture and pre-training methods
of DeBERTaV3 (He et al., 2021). We comprehensively evaluate and compare our model with competitive baselines, i.e., the previous SOTA models
PhoBERT, ViT5, and the multilingual model XLMR on three Vietnamese downstream tasks, including
POS tagging, NER, and QA. In this work, we focus
on two main categories of QA: Machine Reading
Comprehension (MRC) and Open-domain Question Answering (ODQA). The experiment results
show the performance of our model surpasses all
baselines on all tasks. Our main contributions are
summarized as follows:
• We present and implement ViDeBERTa
with three versions: ViDeBERTaxsmall,
ViDeBERTabase, and ViDeBERTalarge which
are the improved large-scale monolingual
language models pre-trained for Vietnamese
based on the DeBERTa architecture and pretraining procedure.
• We also conduct extensive experiments to verify the performance of our pre-trained models
compared to previous strong models in terms
of Vietnamese language modeling. Our empirical results demonstrated the state-of-the-art
(SOTA) results on Vietnamese downstream
tasks: POS tagging, NER, and QA, thus confirming the effectiveness of our improved pretrained language model for Vietnamese.
• Our model, ViDeBERTa, which works with
huggingface and transformers, is available to
the public. We expect that ViDeBERTa will
be an effective pre-trained model for many
NLP applications and research in Vietnamese
and other low-resource languages.
2 Related work
Pre-trained language models for Vietnamese.
PhoBERT (Nguyen and Nguyen, 2020) is the first
large-scale PLM for Vietnamese, which has the
same architecture as BERT (Devlin et al., 2019)
and the same pre-training approach as RoBERTa
(Liu et al., 2019) for more robust performance. This
model was trained on a Vietnamese Wikipedia corpus of 20GB word-level texts and produced SOTA
results on Vietnamese understanding tasks such as
POS, NER, Dependency parsing, and NLI. Following PhoBERT, ViBERT (Tran et al., 2020) and ViELECTRA are public monolingual language models for Vietnamese based on BERT and ELECTRA
pre-training techniques (Clark et al., 2020) that are
pre-trained on syllable-level Vietnamese textual
data. Recent works such as BARTpho (Tran et al.,
2021) and ViT5 (Tran et al., 2020) are pre-trained
for Vietnamese text summarization.
Fine-tuning tasks. This work utilizes three Vietnamese natural language understanding (NLU)
tasks, including POS tagging, NER, and QA, for
fine-tuning and evaluating our model’s performance. For POS tagging and NER, PhoBERT still
produces better results than ViELECTRA, PhoNLP,
and ViT5 (Nguyen and Nguyen, 2020, 2021; Phan
et al., 2022). While early QA (Voorhees et al.,
1999; Brill et al., 2002; Ferrucci et al., 2010) systems were commonly complex and had many parts,
MRC models have evolved and now suggest a simpler two-stage retriever-reader framework (Chen
et al., 2017). A context retriever first selects a small
subset of passages where some of them contain
the answer to the question then a machine reader
can carefully review the retrieved contexts and determine the correct answer. The tasks based on
QA have gained much attention in recent years in
the Vietnamese natural language processing and
computational linguistics community. However, to
the best of our knowledge, there is only the work
(Van Nguyen et al., 2022) that proposes the first
Vietnamese retriever-reader QA system employing
a transformer-based model (XLM-R) evaluated on
the ViQuAD corpus (Nguyen et al., 2020).
3 ViDeBERTa
3.1 Pre-training data
In this work, we use a large corpus CC100
Dataset of 138GB uncompressed texts (Monolingual Datasets from Web Crawl Data) (Conneau
et al., 2020) as a pre-training dataset. This corpus
includes data for romanized languages and monolingual data for more than 100 languages.
According to Nguyen and Nguyen (2020); Tran
et al. (2021), pre-trained language models trained
on word-level data can perform better than those
trained on syllable-level data for word-level Vietnamese NLP tasks. As a result, we perform word
1072
and sentence segmentation using a Vietnamese
toolkit PyVi 1 on the pre-training dataset. After
that, we use a pre-trained SentencePiece tokenizer
from DeBERTaV3 (He et al., 2021) to segment
these sentences with sub-word units, which have a
vocabulary of 128K sub-word types.
3.2 Model Architecture
Our model, ViDeBERTa, follows the DeBERTaV3
architecture by He et al. (2021), which is trained using the self-supervise learning objectives of MLM
and RTD task and a new weight-sharing GradientDisentangled Embedding Sharing (GDES) to enhance the performance of the model. We present
three versions of our model, ViDeBERTaxsmall,
ViDeBERTabase, and ViDeBERTalarge with 22M,
86M, and 304M backbone parameters, respectively.
The details of our model architecture hyperparameters are listed in Table 1.
Table 1: Statistic of our model hyper-parameters. #layer
and #heads denote the numbers of layers and attention
heads of ViDeBERTa model versions, respectively.
Model #layers #heads hidden size
ViDeBERTaxsmall 6 12 768
ViDeBERTabase 12 12 768
ViDeBERTalarge 24 12 1024
3.3 Optimization
We employ our model based on the DeBERTaV3
implementation from (He et al., 2021). We use
Adam (Kingma and Ba, 2015) as the optimizer
with weight decay (Loshchilov and Hutter, 2018)
and use a global batch size of 8,192 across 32 A100
GPUs (80GB each) and a peak learning rate of 6e4 for both ViDeBERTaxsmall and ViDeBERTabase,
while peak learning rate of 3e-4 was used for
ViDeBERTalarge. We pre-train ViDeBERTaxsmall
and ViDeBERTabase for 500k training iterations
and ViDeBERTalarge for 250k training iterations.
4 Experiments and Results
4.1 POS tagging and NER
4.1.1 Experimental setup
For POS tagging and NER tasks, we use standard
benchmarks of the VLSP POS tagging dataset 2
and the PhoNER dataset (Truong et al., 2021).
1
https://pypi.org/project/pyvi/
2
https://vlsp.org.vn/vlsp2013/eval/ws-pos
We follow the procedure in Devlin et al. 2019;
Nguyen and Nguyen 2020 to fine-tune our pretrained model for POS tagging and NER tasks. In
particular, a linear layer for prediction is appended
on top of our model architecture (the last Transformer layer). We then use Adam (Kingma and Ba,
2015) to optimize our model for fine-tuning with
a fixed learning rate of 1e-5 and batch size of 16
(He et al., 2021). The final results for each task and
each dataset are averaged and reported over five
independent runs with different random seeds.
We compare the performance of ViDeBERTa
models with the solid baselines, including
PhoBERT, XLM-R, and ViT5, for these tasks.
Here, XLM-R is a multilingual masked language
model pre-trained on 2.5 TB of CommmonCrawl
dataset of 100 languages, which includes 137GB
of Vietnamese texts.
4.1.2 Main results
Model
POS NER MRC
Acc. F1 F1
XLM-Rbase 96.2
† _ 82.0
‡
XLM-Rlarge 96.3
† 93.8
⋆ 87.0
‡
PhoBERTbase 96.7
† 94.2
⋆ 80.1
PhoBERTlarge 96.8
† 94.5
⋆ 83.5
ViT5base1024−length _ 94.5
⋆ _
ViT5large1024−length _ 93.8
⋆ _
ViDeBERTaxsmall 96.4 93.6 81.3
ViDeBERTabase 96.8 94.5 85.7
ViDeBERTalarge 97.2 95.3 89.9
Table 2: Test results (%) for three tasks POS tagging
(POS for short), NER, and MRC on test sets. Note
that “Acc.” abbreviates the accuracy. †, ⋆, and ‡ denote
scores taken from the PhoBERT paper (Nguyen and
Nguyen, 2020), the ViT5 paper (Phan et al., 2022), and
the ViQuAD paper (Nguyen et al., 2020), respectively.
Table 2 shows the obtained scores of ViDeBERTa compared to the baselines with the highest
reported results. It can be seen clearly that our
model produces significantly better results than the
baselines and achieves new SOTA performance on
both POS tagging and NER tasks.
For POS tagging, ViDeBERTa obtains 0.9% and
0.4% absolute higher accuracy than the large-scale
multilingual model XLM-R (Nguyen et al., 2020)
and the previous SOTA model PhoBERT (Nguyen
and Nguyen, 2020), respectively . Table 2 also
shows our ViDeBERTaxsmall obtains 96.4% accuracy that are better than the baseline XLM-Rlarge
1073
and ViDeBERTabase obtains 96.8% that are competitively the same as the PhoBERTlarge.
For NER, our ViDeBERTalarge achieves F1
score at 95.3% and improves 0.8% absolute
higher score than the previous SOTA models
ViT5base1024−length and PhoBERTlarge. Furthermore, ViDeBERTalarge and ViDeBERTabase preform 1.5% and 0.7% absolute higher scores than
the baseline XLM-Rlarge on the PhoNER corpus.
4.2 Question Answering
4.2.1 Experimental setup
For QA, we evaluate our model on two main tasks:
MRC and ODQA. For ODQA, we propose a new
framework ViDeBERTa-QA, that uses a BM25
(Robertson et al., 2009) as a retriever and ViDeBERTa as a text reader.
Figure 1 depicts an overview of our ViDeBERTa
framework for the Vietnamese Open-domain Question answering task. The statistics of the ViQuAD
dataset used for the task, which is introduced by
Nguyen et al. (2020), are summarized in Table 3.
Corpus #article #passage #question
Train 138 4,101 18,579
Dev 18 515 2,285
Test 18 493 2,21
Full 174 5,109 23,074
Table 3: Statistics of the ViQuAD dataset for QA. “#article”, “#valid”, and “#test” denote the number of articles,
passages, and questions in the ViQuAD, respectively.
We compare ViDeBERTa to the best model
XLM-R (Nguyen et al., 2020) and PhoBERT 3
for
Vietnamese MRC. We also compare our framework, ViDeBERTa-QA, to strong baselines DrQA
(Chen et al., 2017), BERTserini (Yang et al.,
2019a), and the first Vietnamese ODQA system
XLMRQA (Van Nguyen et al., 2022)) that uses
XLM-Rlarge as a reader. We use the ViQuAD corpus introduced by Nguyen et al. (2020) for assessing these tasks. ViQuAD is a Vietnamese corpus
that comprises over 23k triples and each triple includes a question, its answer, and a passage containing the answer.
Similar to POS tagging and NER, we use Adam
(Kingma and Ba, 2015) as an optimizer with a learning rate of 2e-5 and a batch size of 16. We report
3We carefully fine-tune PhoBERT for the MRC task following the fine-tuning approach that we use for ViDeBERTa.
the final results as an average over five independent
runs with different random seeds.
4.2.2 Main results
Table 2 presents the results obtained by ViDeBERTa and two baselines XLM-R (reported by
Nguyen et al. (2020)) and PhoBERT for MRC on
ViQuAD corpus. We find that our ViDeBERTa performance outperforms both XLM-R and PhoBERT
in terms of F1 score.
In particular, the previous SOTA model
XLM-Rlarge for Vietnamese MRC obtains 87%.
Clearly, ViDeBERTa helps boost the XLM-R with
about 2.9% absolute improvement, obtaining a
new SOTA result at 89.9%. In addition, both
versions ViDeBERTabase and ViDeBERTalarge
also outperform PhoBERTbase and PhoBERTlarge
by large margins, respectively. Especially,
ViDeBERTaxsmall (22M parameters) produces
1.2% absolute higher score than PhoBERTbase
(135M parameters) and ViDeBERTabase (86M parameters) produces 2.2% absolute higher score
than PhoBERTlarge (370M parameters) but uses
far fewer parameters than PhoBERT.
For ODQA, Table 4 shows the obtained F1
scores for ViDeBERTa-QA and its baselines on
the test set. Obviously, ViDeBERTa-QA achieves
better scores than the previous SOTA XLMRQA,
BERTsini, and DrQA at the top k passages, selected by retrievers, is 10 and 20. In particular,
ViDeBERTa-QA performs 0.85% (at k = 20) and
0.4% (at k = 10) absolute higher scores than the
previous SOTA system. At smaller k (= 1, 5), ViDeBERTa performs better BERTserini and DrQA by a
large margin; however, XLMRQA does better than
ViDeBERTa-QA.
Model
Top k selected passages
1 5 10 20
DrQA [*] 37.86 37.86 37.86 37.86
BERTserini [*] 55.55 58.30 57.98 58.09
XLMRQA [*] 61.83 64.99 64.49 64.49
ViDeBERTaxsmall 52.76 56.24 56. 93 57.40
ViDeBERTabase 58.55 61.37 61.89 62.43
ViDeBERTalarge 61.23 63.57 64.89 65.34
Table 4: Test scores (F1 in %) for ODQA on ViQuAD
corpus with different k values. Note that [*] indicates
the results reported following Van Nguyen et al. (2022).
4.3 Discussion
According to the results on both downstream tasks
of POS tagging and NER in Table 2, we find that
1074
Retriever
(BM25)
Reader
(ViDeBERTa)
Corpus Top k passages
reader
score
retriever
score
Answer: 1911
Question: Chủ tịch Hồ Chí Minh ra đi tìm
đường cứu nước vào năm nào?
(Which year did President Ho Chi Minh leave
the country to find a way to save the nation?)
Figure 1: An overview of ViDeBERTa-QA framework for Vietnamese Open-domain Question Answering task.
ViDeBERTaxsmall (86M) with fewer parameters
(i.e. only about 15% of XLM-Rlarge 560M and
25% of PhoBERTlarge 370M) but still performs
slightly better than XLM-Rlarge and competitively
the same as the previous SOTA PhoBERTlarge.
One possible reason is that our model inherits
the robustness of DeBERTaV3 architecture and
pre-training techniques, which are demonstrated
superior performance by He et al. (2020, 2021).
Moreover, using more high-quality pre-training
data (138GB) can help ViDeBERTa significantly
improve its performance compared to PhoBERT
(using 20GB).
For Vietnamese QA, the results on the MRC task
show that ViDeBERTa outperforms PhoBERT by
a large margin. It is worth noting that PhoBERT
set a maximum length of 256 subword tokens for
both versions while ViDeBERTa set a larger one of
512. As a result, our models are more scalable than
PhoBERT for long contexts. The results obtained
by ViDeBERTa-QA on ODQA also suggest that
our framework achieves the best performance with
large top k passages selected by the retriever (i.e.
k = 10, 20).
5 Conclusion
In this paper, we have introduced ViDeBERTa, a
new pre-trained large-scale monolingual language
model for Vietnamese. We demonstrate the effectiveness of our ViDeBERTa by showing that ViDeBERTa with fewer parameters performs better than
the recent strong pre-trained language models as
XLM-R, PhoBERT, and ViT5, and achieves SOTA
performances for three downstream Vietnamese
language understanding tasks, including POS tagging, NER, and especially QA. We hope that our
public ViDeBERTa model will boost ongoing NLP
research and applications for Vietnamese and other
low-resource languages.
Limitations
While we have shown that ViDeBERTa can achieve
state-of-the-art performance on a variety of NLP
tasks for Vietnamese, we believe that more analyses
and ablations are required to better understand what
facets of ViDeBERTa contributed to its success and
what knowledge of Vietnamese that ViDeBERTa
captures. We leave these further explorations to
future work.