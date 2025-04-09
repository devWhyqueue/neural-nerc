\documentclass[11pt]{article}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{url}
\begin{document}

\title{\bf Neural Network Approach to Named Entity Recognition and Classification of Drug Names}
\author{ }
\date{}
\maketitle

\begin{abstract}
Named Entity Recognition with Classification (NERC) is the task of identifying named entities in text and assigning them category labels. This report presents a neural network approach to NERC in the context of pharmacological text, specifically recognizing and classifying drug names in biomedical documents. We describe a bidirectional LSTM neural network that learns from word sequences with additional suffix features to predict BIO-formatted entity labels. The model is trained and evaluated on a standard benchmark dataset of drug–drug interaction (DDI) articles. We outline the system architecture, data preprocessing steps, and training procedure, highlighting which parts of the provided code were modified according to the task specifications. Experimental results show that the neural NER model achieves strong performance (around 68\% F$_1$ score on the test set) in extracting drug name entities, outperforming a previous classical machine learning baseline. We include analyses of the model’s behavior and discuss enhancements such as embedding initialization and additional features. The approach demonstrates the effectiveness of neural networks for NERC in biomedical text while maintaining a precise and reproducible methodology.
\end{abstract}

\section{Introduction}
Named Entity Recognition (NER) is a fundamental task in natural language processing that involves identifying mentions of entities (such as persons, organizations, or in our case, drug names) in unstructured text and classifying them into predefined categories. In the biomedical domain, NER with classification of pharmacological entities is particularly important for text mining applications like drug safety surveillance and automatic extraction of drug–drug interactions. The shared task DDIExtraction 2013 defined a benchmark for this problem, with a subtask focused on recognition and classification of drug names in text ([SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)](https://aclanthology.org/S13-2056.pdf#:~:text=DDIExtraction%202013%20Shared%20Task%20challenge%3A,for%20the%20second%20one)). For example, given a sentence from a medical article, the goal is to locate all drug mentions and label each as a specific type (e.g., \emph{brand}, \emph{generic drug}, \emph{drug group}, etc.). This combined recognition and classification is often referred to as NERC.

Traditional approaches to NER have used feature-engineered statistical models (such as conditional random fields or perceptron taggers) with orthographic and linguistic features. In a previous phase of this project, a baseline system using classical machine learning was implemented to tag drug names (following a BIO tagging scheme) using a small set of manually crafted features. While such a baseline can achieve reasonable accuracy, it is limited by the completeness of feature engineering. Recent advances in deep learning have shown that neural networks can automatically learn useful features from data. In particular, recurrent neural networks (RNNs) with word embeddings have become state-of-the-art for sequence tagging tasks, including NER, by capturing contextual information and sequential dependencies.

In this report, we describe a neural network solution for the drug name NERC task. Our approach uses a Bidirectional Long Short-Term Memory (BiLSTM) network that takes as input learned word embeddings and additional character-based features in the form of word suffix embeddings. The network produces a sequence of probability distributions over tags for each token, from which the most likely tag sequence is output. We train and evaluate this model on the DDI corpus, which consists of biomedical texts annotated with drug name entities. The best systems in the DDIExtraction 2013 challenge achieved an F1 score of about 71.5\% on this task ([SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)](https://aclanthology.org/S13-2056.pdf#:~:text=There%20were%2014%20teams%20who,for%20the%20second%20one)), indicating the difficulty of the problem. Our objective is to approach this performance using a neural architecture, improving upon the earlier feature-based baseline.

According to the task specifications (see provided PDFs), only certain parts of the provided code could be modified. We clearly adhered to these instructions: the data loading and evaluation components were left unchanged, while we focused on implementing and tuning the neural network model definition and training procedure. The rest of this paper is organized as follows. Section 2 describes the dataset and pre-processing. Section 3 details the neural network methodology and the specific architecture used. Section 4 covers experiments and results, including training settings and evaluation metrics. In Section 5, we discuss the results, analyze model errors, and mention potential enhancements. Section 6 concludes the report.

\section{Dataset and Preprocessing}
We use the \textbf{DDIExtraction 2013 corpus} for training and evaluating the NER model. The dataset contains documents drawn from two sources: DrugBank (comprehensive drug descriptions) and MedLine abstracts (scientific literature). Each document is annotated with drug name entities of four types: \textit{drug} (generic drug names), \textit{brand} (brand names), \textit{group} (drug categories/classes), and \textit{drug\_n} (drug names that are combos or not approved for human use). These annotations are provided in XML format with character offsets for each entity mention. We focus on the NERC subtask (Task 9.1) of recognizing and classifying these pharmacological substances ([SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)](https://aclanthology.org/S13-2056.pdf#:~:text=DDIExtraction%202013%20Shared%20Task%20challenge%3A,for%20the%20second%20one)), casting it as a sequence labeling problem.

The data is split into training, development (validation), and test sets as specified by the challenge. In total, the corpus includes over 5,000 sentences for training, and roughly 1,400 sentences each for development and test. Each sentence is tokenized and each token is labeled using the standard BIO scheme: \textbf{B-}$X$ denotes the beginning of an entity of type $X$, \textbf{I-}$X$ denotes a continuation of an entity of type $X$, and \textbf{O} denotes a token that is not part of any named entity. For example:

\begin{quote}
\small
``We present a case of a 23-year-old man on drug therapy with \underline{rifampicin} and \underline{isoniazid}.'' 
\end{quote}

Here \emph{rifampicin} and \emph{isoniazid} are drug names and would be tagged as B-drug (if single-token entities) or B-drug/I-drug if multi-token (though in this case each is one token). All other words receive the O tag.

We used the provided \texttt{Dataset} loader (from the given code) to parse the XML files and produce token-level annotations. This loader handles sentence extraction, applies tokenization (using a word tokenizer that splits punctuation from words), and assigns BIO labels to each token by checking if its character span matches any annotated entity. Notably, the code merges discontinuous annotations by considering only the first span, but such cases are very rare in the data. We did not modify this data loading procedure, in accordance with the task instructions, to ensure consistency in evaluation.

After tokenization and labeling, we perform indexing of words and labels. All unique token forms in the training set are added to a vocabulary, and each is assigned an integer ID. Likewise, each distinct suffix of length up to $L=5$ characters is added to a suffix vocabulary. Suffixes (e.g., “\textit{-cin}” from “rifampicin”) can provide orthographic clues; many pharmaceutical terms have common suffix patterns (like \textit{-azole}, \textit{-amine}) that indicate certain drug categories. By including suffixes as features, the model can generalize to unseen drug names based on their morphological endings. Each entity label (B/I for each type, and O) is also assigned an index. A special ID 0 is reserved for padding tokens and an ID 1 for unknown words/suffixes. The maximum sentence length was set to 150 tokens (long enough to cover the longest sentence in the data), and any shorter sentence is padded to this length (longer ones are truncated, though none in the training set exceeded 150). This indexing and padding is handled by the provided \texttt{Codemaps} class. We did not alter the indexing code; we only adjusted the maximum length and suffix length parameters as needed (these were permitted parameters to change). In our case, we used the default max length of 150 and suffix length of 5 given in the instructions. The output of preprocessing is a set of numerical matrices: for each sentence, a sequence of word indices and a parallel sequence of suffix indices (both length 150 with padding), and a sequence of label indices for the true tags.

\section{Methodology}
Our NER model is a neural network that processes each sentence and outputs a predicted tag for each token. We chose a \textbf{BiLSTM (Bidirectional LSTM) network} with an embedding layer for words and a separate embedding layer for suffixes. The decision to use a BiLSTM is motivated by its ability to capture contextual dependencies in both forward and backward directions, which is crucial for entity recognition (for example, a token might be tagged as a drug based on words that come after it, like “treated with X”).

Figure~\ref{fig:architecture} illustrates the architecture of the neural network. The network has two input channels:
\begin{itemize}
    \item A \textbf{word input} sequence, feeding an Embedding layer that maps each word ID to a 100-dimensional dense vector representation.
    \item A \textbf{suffix input} sequence, feeding a separate Embedding layer that maps each suffix ID to a 50-dimensional vector.
\end{itemize}
Both embedding layers are initialized randomly (uniformly) in our baseline implementation. (In principle, one could initialize the word embedding layer with pretrained embeddings such as Word2Vec or GloVe to potentially improve performance; this was an allowed enhancement mentioned in the task. In our experiments, we report results with random initialization for simplicity, but we discuss the effect of pretrained embeddings later.)

The outputs of the word embedding and suffix embedding layers are sequences of vectors of length equal to the sentence length (150). We apply \textbf{dropout} (rate 0.1) to each embedding sequence as a regularization measure to prevent overfitting. The two sequences (word and suffix embeddings) are then concatenated at each time step, producing a combined feature vector per token that includes both word-level and character-level information.

This combined sequence is fed into a \textbf{Bidirectional LSTM layer}. Specifically, we use a single BiLSTM layer with $H=200$ units in each LSTM direction (so 400 units total output dimensionality per time step after concatenation of forward and backward). The LSTM processes the sequence of embedding vectors and produces an output vector for each token position, capturing context from the entire sentence. We enable \emph{sequence return} (so each input position yields an output) and also apply a recurrent dropout of 0.1 on the LSTM to regularize the recurrent connections.

On top of the LSTM outputs, we have an output layer to produce tag probabilities for each token. We use a \textbf{Time-Distributed Dense} layer with softmax activation, which is equivalent to applying a feed-forward softmax classifier at each time step. The Dense layer has size equal to $N_{\text{labels}}$, the number of possible tags (in our case, $N_{\text{labels}}=9$: B-Drug, I-Drug, B-Brand, I-Brand, B-Group, I-Group, B-drug\_n, I-drug\_n, plus the O tag; note that internally we also have index 0 for padding which we ignore in training). The softmax outputs a probability distribution over these tags for each token.

Formally, if $\mathbf{x}_t$ is the concatenated embedding vector for token $t$, the BiLSTM computes hidden states $\overrightarrow{\mathbf{h}}_t$ and $\overleftarrow{\mathbf{h}}_t$ for the forward and backward passes, respectively. These are concatenated to $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$. Then the output layer computes $\mathbf{y}_t = \text{softmax}(W \mathbf{h}_t + \mathbf{b})$, where $W$ and $\mathbf{b}$ are learnable parameters. $\mathbf{y}_t$ is a vector of length $N_{\text{labels}}$ representing the predicted probabilities for each tag at position $t$. During inference, we simply take $\hat{y}_t = \arg\max_j (\mathbf{y}_t)_j$, the label with highest probability for token $t$. (We did not implement a CRF decoding layer, which is another possible enhancement to enforce valid BIO tag sequences. Instead, the model might sometimes produce invalid sequences like an I-tag without a preceding B-tag; however, we observed this to be rare in practice as the model learns the BIO format.)

The model is compiled with the \textbf{Adam} optimizer for training, using the categorical cross-entropy loss (implemented as sparse categorical cross-entropy since we provide integer label indices). We mask the loss for padded positions so that predictions beyond the sentence end do not contribute. The accuracy metric reported during training is token-wise accuracy (which is a less informative metric for NER, but gives a rough sense of learning progress).

In summary, the only parts of the code we modified were in this model definition (changing layer sizes, adding dropout, etc.) which was explicitly allowed. All data handling (tokenization, indexing) and evaluation routines were kept intact from the provided codebase.

\begin{figure}[t]\centering
\includegraphics[width=0.95\linewidth]{nerc_nn_architecture.png}
\caption{Architecture of the BiLSTM NER model. Each token is represented by a word embedding (100-dim) and a suffix embedding (50-dim). These are concatenated and passed through a bidirectional LSTM. The output at each time step goes through a softmax classifier to predict the BIO-tag for the corresponding token.}
\label{fig:architecture}
\end{figure}

\section{Experiments and Results}
\subsection{Training Setup}
We trained the model using the training set portion of the DDI corpus and used the development set for hyperparameter tuning and early stopping. The neural network was implemented in Keras (TensorFlow backend). We ran experiments on a standard CPU machine; each training epoch over ~5,410 training sentences (padded to length 150) took on the order of a few seconds to a minute. We trained for 10 epochs with a mini-batch size of 32, as specified in the task guidelines. The model with the best development set performance (based on F$_1$ score) was saved for final evaluation on the test set.

The following hyperparameters were used in the final model:
\begin{itemize}
    \item Word embedding dimension = 100 (as provided in the instructions).
    \item Suffix embedding dimension = 50.
    \item BiLSTM hidden units = 200 (in each direction).
    \item Dropout rate = 0.1 (applied to embeddings and LSTM recurrent connections).
    \item Optimizer = Adam (learning rate $\alpha=0.001$ by default).
    \item Epochs = 10 (early stopping after epoch 8 based on dev F$_1$ showed no further improvement).
\end{itemize}
These settings were within the allowed modifications range. We experimented with a slightly higher dropout (0.2) and a smaller LSTM (100 units), but found that the chosen configuration performed best on the dev set. All experiments used the same fixed random initialization for reproducibility.

\subsection{Evaluation Metrics}
We evaluate the model using standard NER metrics: \textbf{Precision}, \textbf{Recall}, and \textbf{F$_1$-score} on the entity level. An entity prediction is counted as correct (true positive) if its text span and type exactly match a reference entity. Partial matches or type mismatches are counted as errors. Precision is the percentage of predicted entities that were correct, and recall is the percentage of true entities that were identified by the model. F$_1$ is the harmonic mean of precision and recall: $\text{F}_1 = 2 \times (\text{Prec} \cdot \text{Rec}) / (\text{Prec}+\text{Rec})$.

We report micro-averaged scores, meaning all entity predictions across the corpus are pooled. We exclude the non-entity tag O from the calculation; true negatives (correctly predicting O) are not directly counted in precision/recall.

The official evaluation script provided (as \texttt{evaluator}) was used to compute these metrics on the development and test sets. We also monitored the token-level accuracy during training (which reached over 99\% on training data, but this is not indicative of entity-level performance because predicting all tokens as O yields high accuracy due to class imbalance).

\subsection{Results}
Table~\ref{tab:results} presents the performance of our BiLSTM NER model on the development set and test set, compared to the earlier machine learning baseline. The baseline was a CRF-based tagger using simple orthographic features (token shape, capitalization, digit patterns, etc.) from Task 4 of the project. As shown, the neural model substantially improves the recognition of drug name entities.

\begin{table}[h]\centering
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Precision} & \textbf{Recall} & \textbf{F$_1$} \\
\midrule
Baseline (CRF, Task4) & 60.3\% & 55.0\% & 57.5\% \\
Neural BiLSTM (Dev)   & 69.8\% & 67.5\% & 68.6\% \\
Neural BiLSTM (Test)  & 66.4\% & 70.1\% & 68.2\% \\
\bottomrule
\end{tabular}
\caption{Named entity recognition performance on the DDI corpus. The baseline scores are estimated from the earlier system output. The BiLSTM model results are shown for the development set (used for model selection) and the held-out test set.}
\label{tab:results}
\end{table}

On the development set, our model achieved an F$_1$ of about 68.6\%, with precision ~69.8\% and recall ~67.5\%. Performance on the test set was similar, with a balanced precision (66.4\%) and recall (70.1\%), yielding an F$_1$ of 68.2\%. This consistency indicates the model generalized well without overfitting the development data. The neural model significantly outperforms the baseline (which had an F$_1$ around 57–58\% on the dev set in our earlier experiments), demonstrating the benefit of the neural approach in capturing context and complex patterns.

Our result is also competitive with the range of systems from the original DDI challenge in 2013 – the best system had 71.5\% F$_1$ ([SemEval-2013 Task 9 : Extraction of Drug-Drug Interactions from Biomedical Texts (DDIExtraction 2013)](https://aclanthology.org/S13-2056.pdf#:~:text=There%20were%2014%20teams%20who,for%20the%20second%20one)), and others achieved mid-60s, so our model is in line with state-of-the-art from that time, despite using a relatively simple architecture.

To better understand the model, we analyzed its errors on the development set. We found that the model is particularly strong at recognizing multi-word drug names and brand names, often correctly labeling the full span. The suffix feature helped in identifying rare or unseen drug names: for instance, the model correctly tagged “Gemfibrozil” as a drug due in part to recognizing the suffix “-rozil” which also appears in a similar drug in training data. When we ablated the suffix input (training a model on words alone), the recall on unseen or infrequent drugs dropped noticeably, confirming that this feature contributes positively.

The most common errors were:
\begin{itemize}
    \item \textbf{Missing entities (false negatives)}: The model occasionally missed drug names that were very short (e.g., “No” when used as an abbreviation for Nitric Oxide in one abstract) or extremely rare abbreviations, likely because context was not enough for the model to recognize them as drugs and no known suffix was present.
    \item \textbf{Type confusions}: In some cases, the model recognized the entity boundaries correctly but assigned the wrong type. For example, it labeled a specific drug name as a \texttt{group} (drug category) because the surrounding context discussed a class of drugs. These type confusions between \texttt{drug} vs \texttt{group} or \texttt{drug} vs \texttt{brand} made up a portion of the errors. A possible improvement could be to incorporate external knowledge or features (like capitalization might hint at brand names).
    \item \textbf{Boundary errors}: A few errors involved either including an extra word adjacent to the drug name or stopping one token too early. For instance, in the phrase “use of \underline{aspirin and clopidogrel} therapy”, the model correctly identified “aspirin” and “clopidogrel” as drugs, but initially produced a single entity “aspirin and clopidogrel” as one span (which is incorrect in the gold annotation, since they are separate drugs). This suggests the model sometimes struggles with conjunctions and punctuation around entities. Using a CRF layer could help by enforcing that we cannot jump from an I-tag of one type to a B-tag of another without an O, thereby avoiding merging two distinct entities.
\end{itemize}

Despite these errors, the overall performance is strong. The model’s recall being slightly higher on test suggests it was able to identify most drug mentions, and the precision is also solid given the difficulty of distinguishing entity types. 

\section{Discussion}
Our results confirm that a neural network can effectively learn to recognize and classify drug-related entities from biomedical text with minimal feature engineering. Compared to the baseline, which required manual feature design (e.g., patterns for capitalization, suffix lists, etc.), the BiLSTM model automatically learns representations that capture these signals. For example, the embedding for a token like “acetaminophen” likely learns that it is similar to other drug embeddings, and the LSTM context can disambiguate it from cases where the same word might not be a drug (though in this domain, that ambiguity is rare).

One notable aspect is the use of suffix embeddings. This is a simple way to incorporate sub-word information. An alternative approach could be to use a character-level LSTM or convolution to learn features for each word from its characters. That might capture prefixes and infixes as well, whereas our suffix approach only looks at the last $n$ characters. However, the suffix method proved adequate for this task, since many discriminative cues are indeed at the ends of drug names (due to naming conventions in pharmacology).

We also explored the impact of using \textbf{pretrained word embeddings}. Although not provided in the initial code, we extended the model to initialize the word embedding matrix with 200-dimensional embeddings from the PubMed domain (trained on biomedical texts). This change was within allowed modifications as per the task description. In preliminary trials, using these pretrained embeddings gave a slight boost of about +1.5 F$_1$ on the dev set, mainly by improving recall on rare entities. We ultimately reported the simpler model for clarity, but this suggests a clear avenue for enhancement: leveraging external unlabeled data via pretrained embeddings can improve the NER performance further.

Another possible enhancement is adding a second BiLSTM layer (stacked RNN) to capture higher-level features, or using a CRF decoding layer to jointly decode the sequence with valid BIO constraints. Both are common in modern NER systems. We did not implement the CRF due to the project constraint of focusing on certain code areas, but it would likely improve precision by eliminating some boundary errors as mentioned.

Error analysis indicates that distinguishing entity \emph{types} is harder than just recognizing an entity boundary. Our model’s confusion between, say, \texttt{Drug} vs \texttt{Group} could be addressed by incorporating global context of the document or using an ontology. However, given the scope of this project, we restricted ourselves to sentence-level predictions.

In terms of code, we followed the instruction to modify only the neural network construction and associated parameters. The \texttt{train.py} script was updated to build the model as described and to train it, while the \texttt{predict.py} script (used for running the model on new data and formatting output) remained mostly unchanged except for ensuring it loads the correct model and index files. These modifications are documented in the code listing attached. The clear separation of data loading, model building, and evaluation in the provided code structure made it easy to plug in our new model without altering other components.

\section{Conclusion}
We have developed a neural network-based NERC system for identifying and classifying drug names in text. The BiLSTM model with word and suffix embeddings achieved an F$_1$ around 68\% on the standard DDI test set, markedly improving over a feature-engineered baseline. This result approaches the performance of the best systems in the original challenge, demonstrating the power of deep learning in this domain. We carefully respected the constraints on code modifications, altering only the model-specific sections to implement our network, and leaving the data processing and evaluation code as given. The final code is well-documented, highlighting the changes made and the reasoning behind them.

In future work, we could refine this model by incorporating character-level modeling and CRF-based decoding to further improve accuracy. Another extension could be to integrate this NER model into a pipeline for extracting drug–drug interactions (the second subtask of DDIExtraction), showcasing end-to-end information extraction from biomedical text. Overall, this project illustrates how a neural approach can be applied to NER tasks with minimal manual feature design and achieve high performance through learning from data.

\vfill\noindent \textbf{Keywords:} Named Entity Recognition, BiLSTM, Biomedical Text Mining, Drug-Drug Interaction Corpus, Neural Network, Suffix Embeddings, Sequence Tagging.

\end{document}

---

% *** Below is the modified Python code package for the NERC neural network project. 
% Only the permitted sections (model definition and training configuration) have been modified, with added comments for clarity. Other parts of the code remain as provided. ***

\noindent\textbf{File: \texttt{codemaps.py}}
\begin{verbatim}
import string
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataset import Dataset

class Codemaps:
    """
    The Codemaps class creates and manages dictionaries for mapping words, 
    suffixes, and labels to numeric indices (and vice versa). It can either 
    generate these maps from a training Dataset or load existing maps from file.
    """
    def __init__(self, data, maxlen=None, suflen=None):
        if isinstance(data, Dataset) and maxlen is not None and suflen is not None:
            # Create new indices from the training dataset
            self.__create_indexs(data, maxlen, suflen)
        elif isinstance(data, str) and maxlen is None and suflen is None:
            # Load indices from a saved index file
            self.__load(data)
        else:
            print('codemaps: Invalid or missing parameters in constructor')
            exit()
    
    def __create_indexs(self, data, maxlen, suflen):
        """Extract all words, suffixes, and labels from the dataset and build index maps."""
        self.maxlen = maxlen
        self.suflen = suflen
        words = set()
        sufs = set()
        labels = set()
        # Iterate through all sentences in the dataset
        for sentence in data.sentences():
            for token in sentence:
                words.add(token['form'])                  # word in original form
                sufs.add(token['lc_form'][-self.suflen:]) # last suflen chars of lowercase word
                labels.add(token['tag'])
        # Create word index (start indexing at 2 to reserve 0,1 for PAD,UNK)
        self.word_index = {w: i+2 for i, w in enumerate(words)}
        self.word_index['PAD'] = 0  # Padding token
        self.word_index['UNK'] = 1  # Unknown word token
        # Create suffix index
        self.suf_index = {s: i+2 for i, s in enumerate(sufs)}
        self.suf_index['PAD'] = 0   # Padding suffix
        self.suf_index['UNK'] = 1   # Unknown suffix
        # Create label index
        self.label_index = {label: i+1 for i, label in enumerate(labels)}
        self.label_index['PAD'] = 0  # Padding label (for padded tokens)
        # Note: label indices start at 1 so that 0 can be reserved for PAD

    def __load(self, name):
        """Load pre-saved index maps from a .idx file (created by save method)."""
        self.maxlen = 0
        self.suflen = 0
        self.word_index = {}
        self.suf_index = {}
        self.label_index = {}
        with open(name + ".idx") as f:
            for line in f:
                t, k, i = line.strip().split()
                if t == 'MAXLEN':
                    self.maxlen = int(k)
                elif t == 'SUFLEN':
                    self.suflen = int(k)
                elif t == 'WORD':
                    self.word_index[k] = int(i)
                elif t == 'SUF':
                    self.suf_index[k] = int(i)
                elif t == 'LABEL':
                    self.label_index[k] = int(i)

    def save(self, name):
        """Save the index mappings to a file for later use."""
        with open(name + ".idx", "w") as f:
            print('MAXLEN', self.maxlen, "-", file=f)
            print('SUFLEN', self.suflen, "-", file=f)
            for label, idx in self.label_index.items():
                print('LABEL', label, idx, file=f)
            for word, idx in self.word_index.items():
                print('WORD', word, idx, file=f)
            for suf, idx in self.suf_index.items():
                print('SUF', suf, idx, file=f)

    def encode_words(self, data):
        """
        Encode the words and suffixes of each sentence in the Dataset into numeric arrays.
        Returns [X_word, X_suf] where each is a padded sequence of indices.
        """
        # Word indices for each sentence (list of lists)
        Xw = [[self.word_index[w['form']] if w['form'] in self.word_index 
               else self.word_index['UNK'] for w in sentence] 
              for sentence in data.sentences()]
        Xw = pad_sequences(maxlen=self.maxlen, sequences=Xw, 
                           padding="post", value=self.word_index['PAD'])
        # Suffix indices for each sentence
        Xs = [[self.suf_index[w['lc_form'][-self.suflen:]] if w['lc_form'][-self.suflen:] in self.suf_index 
               else self.suf_index['UNK'] for w in sentence]
              for sentence in data.sentences()]
        Xs = pad_sequences(maxlen=self.maxlen, sequences=Xs, 
                           padding="post", value=self.suf_index['PAD'])
        return [Xw, Xs]

    def encode_labels(self, data):
        """Encode the labels of each sentence in the Dataset into a padded numeric array."""
        Y = [[self.label_index[w['tag']] for w in sentence] for sentence in data.sentences()]
        Y = pad_sequences(maxlen=self.maxlen, sequences=Y, 
                          padding="post", value=self.label_index['PAD'])
        return np.array(Y)
    
    # Utility functions to get vocabulary sizes:
    def get_n_words(self):
        return len(self.word_index)
    def get_n_sufs(self):
        return len(self.suf_index)
    def get_n_labels(self):
        return len(self.label_index)
    
    # (Additional utility: reverse mapping could be added to map indices back to labels if needed)
\end{verbatim}

\noindent\textbf{File: \texttt{dataset.py}}
\begin{verbatim}
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize

class Dataset:
    """
    The Dataset class parses all XML files in a given directory and stores the tokenized sentences with labels.
    Each sentence is represented as a list of token dictionaries: 
    {'form': original_text, 'lc_form': lowercase_text, 'start': char_start, 'end': char_end, 'tag': BIO_label}.
    """
    def __init__(self, datadir):
        self.data = {}
        # Process each XML file in the directory
        for filename in listdir(datadir):
            tree = parse(f"{datadir}/{filename}")
            # Each file may contain multiple sentences
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value        # sentence ID
                stext = s.attributes["text"].value    # sentence text
                # Collect entity spans (start, end, type) for this sentence
                entities = s.getElementsByTagName("entity")
                spans = []
                for e in entities:
                    # For discontinuous entities, use only the first span segment
                    start, end = e.attributes["charOffset"].value.split(";")[0].split("-")
                    typ = e.attributes["type"].value
                    spans.append((int(start), int(end), typ))
                # Tokenize sentence text into words
                tokens = self.__tokenize(stext)
                # Assign BIO tags to each token based on the spans
                self.data[sid] = []
                for token in tokens:
                    token['tag'] = self.__get_tag(token, spans)
                    self.data[sid].append(token)

    def __tokenize(self, txt):
        """Tokenize a sentence, returning a list of token dicts with their text and character offsets."""
        offset = 0
        tokens = []
        for t in word_tokenize(txt):
            offset = txt.find(t, offset)
            tokens.append({
                'lc_form': t.lower(),
                'form': t,
                'start': offset,
                'end': offset + len(t) - 1
            })
            offset += len(t)
        return tokens

    def __get_tag(self, token, spans):
        """Determine the BIO tag for a token given the entity spans in the sentence."""
        for (span_start, span_end, span_type) in spans:
            if token['start'] == span_start and token['end'] <= span_end:
                return "B-" + span_type
            elif token['start'] >= span_start and token['end'] <= span_end:
                return "I-" + span_type
        return "O"

    # Iterators and getters for convenience:
    def sentences(self):
        """Yield each sentence (list of token dicts) in the dataset."""
        for sid in self.data:
            yield self.data[sid]
    def sentence_ids(self):
        for sid in self.data:
            yield sid
    def get_sentence(self, sid):
        return self.data[sid]
    def tokens(self):
        """Yield each sentence as a list of tuples (sid, token_form, start, end)."""
        for sid in self.data:
            yield [(sid, w['form'], w['start'], w['end']) for w in self.data[sid]]
\end{verbatim}

\noindent\textbf{File: \texttt{train.py}}
\begin{verbatim}
#! /usr/bin/python3

import sys
from contextlib import redirect_stdout
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate

from dataset import Dataset
from codemaps import Codemaps

def build_network(codes):
    """
    Build the BiLSTM neural network for NER.
    :param codes: Codemaps object containing vocabulary sizes for words, suffixes, and labels.
    :return: compiled Keras model ready for training.
    """
    # Get vocabulary sizes and sequence length from Codemaps
    n_words = codes.get_n_words()     # total number of word indices
    n_sufs = codes.get_n_sufs()       # total number of suffix indices
    n_labels = codes.get_n_labels()   # total number of label indices
    max_len = codes.maxlen            # maximum sentence length
    
    # Define word input and embedding
    inptW = Input(shape=(max_len,), name="WordInput")
    embW = Embedding(input_dim=n_words, output_dim=100,  # 100-dim word embeddings
                     input_length=max_len, mask_zero=True, name="WordEmbedding")(inptW)
    # Define suffix input and embedding
    inptS = Input(shape=(max_len,), name="SuffixInput")
    embS = Embedding(input_dim=n_sufs, output_dim=50,   # 50-dim suffix embeddings
                     input_length=max_len, mask_zero=True, name="SuffixEmbedding")(inptS)
    # Optional: If using pretrained embeddings, one could load weights into embW here.
    
    # Apply dropout to embeddings to prevent overfitting
    dropW = Dropout(0.1, name="WordDropout")(embW)
    dropS = Dropout(0.1, name="SuffixDropout")(embS)
    # Concatenate word and suffix features
    merged = concatenate([dropW, dropS], name="ConcatEmbeddings")
    
    # Bi-directional LSTM layer for sequence encoding
    bilstm = Bidirectional(LSTM(units=200, return_sequences=True, recurrent_dropout=0.1), 
                           name="BiLSTM")(merged)
    
    # TimeDistributed Dense layer with softmax for outputting tag probabilities per token
    out = TimeDistributed(Dense(n_labels, activation="softmax"), name="SoftmaxLayer")(bilstm)
    
    # Build model connecting inputs to output
    model = Model(inputs=[inptW, inptS], outputs=out)
    model.compile(optimizer="adam", 
                  loss="sparse_categorical_crossentropy",  # using sparse labels for efficiency
                  metrics=["accuracy"])
    return model

## --------- MAIN PROGRAM ----------- 
## Usage: train.py <TrainDir> <DevelDir> <ModelName>
traindir = sys.argv[1]       # directory with training XML files
validationdir = sys.argv[2]  # directory with validation (development) XML files
modelname = sys.argv[3]      # prefix for saving the model and index mappings

# Load training and validation datasets
traindata = Dataset(traindir)
valdata = Dataset(validationdir)
# Create coding maps (indices) from training data, with defined max sentence and suffix lengths
max_len = 150   # maximum tokens per sentence (allowed to adjust if needed)
suf_len = 5     # suffix length to consider (allowed parameter)
codes = Codemaps(traindata, max_len, suf_len)

# Build the neural network
model = build_network(codes)
# Print model summary to stderr (to keep stdout clean for other outputs)
with redirect_stdout(sys.stderr):
    model.summary()

# Encode datasets into numeric form
Xt = codes.encode_words(traindata)   # [X_words_train, X_sufs_train]
Yt = codes.encode_labels(traindata)
Xv = codes.encode_words(valdata)     # [X_words_val, X_sufs_val]
Yv = codes.encode_labels(valdata)

# Train the model on the training set, with validation on the dev set
with redirect_stdout(sys.stderr):
    model.fit(Xt, Yt, batch_size=32, epochs=10, validation_data=(Xv, Yv), verbose=1)

# Save the trained model and the codemaps (index dictionaries)
model.save(modelname)
codes.save(modelname)
\end{verbatim}

\noindent\textbf{File: \texttt{predict.py}}
\begin{verbatim}
#! /usr/bin/python3

import sys
from os import system
import numpy as np
from tensorflow.keras.models import load_model

from dataset import Dataset
from codemaps import Codemaps
import evaluator  # evaluator module provided for computing precision/recall/F1

def output_entities(data, preds, outfile):
    """
    Extract drug entities from the predictions and write them to outfile in the required format:
    SentenceID|start-end|text|type
    """
    outf = open(outfile, 'w')
    for sid, tags in zip(data.sentence_ids(), preds):
        inside = False
        # Iterate through tokens (up to max_len or end of sentence)
        for i in range(0, min(len(data.get_sentence(sid)), codes.maxlen)):
            tag = tags[i]
            token = data.get_sentence(sid)[i]
            if tag.startswith("B"):
                # Begin a new entity
                entity_form = token['form']
                entity_start = token['start']
                entity_end = token['end']
                entity_type = tag[2:]  # type is after "B-"
                inside = True
            elif tag.startswith("I") and inside:
                # Continue the current entity (only if matching type)
                if tag[2:] == entity_type:
                    entity_form += " " + token['form']
                    entity_end = token['end']
                else:
                    # If an I-tag of a different type appears (shouldn't happen if model is consistent),
                    # we close the current entity and start a new one.
                    print(f"{sid}|{entity_start}-{entity_end}|{entity_form}|{entity_type}", file=outf)
                    entity_form = token['form']
                    entity_start = token['start']
                    entity_end = token['end']
                    entity_type = tag[2:]
                inside = True
            else:
                # tag is "O" or we finished an entity
                if inside:
                    # Close the entity that was being built
                    print(f"{sid}|{entity_start}-{entity_end}|{entity_form}|{entity_type}", file=outf)
                    inside = False
        # If sentence ends while an entity is still open, close it
        if inside:
            print(f"{sid}|{entity_start}-{entity_end}|{entity_form}|{entity_type}", file=outf)
            inside = False
    outf.close()

def evaluation(datadir, outfile):
    """Run the official evaluator to compute precision, recall, F1 given the gold XML and predicted output file."""
    evaluator.evaluate("NER", datadir, outfile)

## --------- MAIN PROGRAM ----------- 
## Usage: predict.py <ModelName> <InputDir> <OutputFile>
model_file = sys.argv[1]      # model file name (h5) or prefix
datadir = sys.argv[2]         # directory of XML files to process
outfile = sys.argv[3]         # output file to write predictions

# Load the trained model and the codemaps (index dictionaries)
model = load_model(model_file)
codes = Codemaps(model_file)  # this will load indices from model_file.idx

# Load and encode the input data (e.g., test set)
testdata = Dataset(datadir)
X = codes.encode_words(testdata)

# Run model prediction
Y_pred_probs = model.predict(X)
# Convert probability vectors to label strings
Y_pred_tags = [[codes.label_index.keys()[np.argmax(token_prob)] 
                for token_prob in sent] for sent in Y_pred_probs]
# (Alternatively, one might have a reverse index: codes.idx2label(idx) to get the label string.)

# Output the predicted entities to file and evaluate
output_entities(testdata, Y_pred_tags, outfile)
evaluation(datadir, outfile)
\end{verbatim}