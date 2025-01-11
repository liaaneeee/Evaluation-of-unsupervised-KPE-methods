# Evaluation of unsupervised Keyphrase extraction (KPE) methods

This repository contains experiments for the evaluation of three unsupervised Keyphrase extraction (KPE) models: [KPMiner](https://doi.org/10.1016/j.is.2008.05.002), [Multipartite Rank (MPRank)](https://doi.org/10.18653/v1/N18-2105) & [EmbedRank](https://doi.org/10.18653/v1/K18-1022). The evaluation makes use of the dataset [SemEval 2010](https://aclanthology.org/S10-1004/). A version of the dataset is publicly available on [Hugging Face](https://huggingface.co/datasets/midas/semeval2010).

## Important note:
Due to dependency conflicts that occured during the installation of required packages for the three KPE models, running the experiments with **Option 2** (please refer to **Usage options** below) requires setting up two separate environments (see Required packages section): one for performing the Keyphrase extraction using EmbedRank (Python 3.6) and one for performing the Keyphrase extraction using MPRank & EmbedRank (Python 3.11). 


## Usage
### Usage options
This repository contains serialized keyphrase extractions from the three models for the evaluation. Therefore, there are two options for running the experiments:

- **Option 1:** Use the serialized data for the evaluation; the notebooks 1 and 2 (see below) contain code for deserializing the extracted keyphrases for evaluation 
- **Option 2:** Perform the extractions from scratch; note that the extraction takes time and that setting up working environments can be a bit cumbersome (please refer to the **Required packages** section of this README)

### Which files do I need for running the experiments?

The experiments for hyperparameter tuning and the evaluation of KPMiner, MPRank and EmbedRank are in the following two Jupyter notebooks:
- 1. `pke/hyperparameter_tuning.ipynb`
- 2. `pke/evaluation.ipynb`


If you choose **Option 2** (see options above), you need to run the `embedrank/embedrank_extraction.ipynb` notebook first before running the above notebooks in order to obtain the keyphrase extractions from EmbedRank. 

## Repository structure
- `embedrank/` - Contains the code for performing Keyphrase extraction with the EmbedRank model.
    - `embedrank_extraction.ipynb` - A Jupyter notebook containing code for the extraction of keyphrases from the SemEval 2010 train & test documents using EmbedRank

    - `launch.py` - This file is taken from the [repository](https://github.com/swisscom/ai-research-keyphrase-extraction) containing the official implementation of EmbedRank, the code belongs to the authors of that repository; the module is needed for launching the EmbedRank model for Keyphrase extraction

    - `data/` - Contains the train and test documents of the SemEval 2010 datasets as TXT-files as well as two TXT-files listing the order of the documents from the Hugging Face version of the dataset (EmbedRank requires Python 3.6 while the Hugging Face datasets library requires Python 3.7+ )


- `pke/` - Contains the code for performing Keyphrase extraction with KPMiner & EmbedRank using the Python-based library [PKE](https://github.com/boudinfl/pke) for Keyphrase extraction; also contains the main experiments for hyperparameter tuning and the evaluation of the three KPE models.

    - `hyperparameter_tuning.ipynb` - A Jupyter notebook containing experiments for hyperparameter tuning 
    - `evaluation.ipynb` - A jupyter notebook containing the main experiments for the evaluation
    - `util/` - A module containing utiliy functions and the implementations of the metrics used for the evaluation
    - `df-semeval2010.tsv.gz` - This file is taken from the [PKE repository](https://github.com/boudinfl/pke) and belongs to the authors of this repository; the file contains document frequencies learned from SemEval 2010 required for the computation of Tf-idf by KPMiner


- `extractions/` - Contains the serialized extracted keyphrases obtained from KPMiner, MPRank and EmbedRank during the experiments in JSON; you can simply deserialize these extractions in order to compute the results for the experiments without performing the extractions yourself


## Required packages

### 1. Set up a virtual environment with Python 3.11:
- Required for running the notebooks in the `pke/`-directory 

1. Create a venv with **Python 3.11** and make sure to activate it 

2. Pip install required packages:
    - [pke 2.0.0](https://github.com/boudinfl/pke/tree/master) 
        - Don't forget to download the English language model for spaCy "en_core_web_sm"
    - [datasets 3.2.0](https://huggingface.co/docs/datasets/installation) 
    - [sentence_transformers 3.3.1](https://sbert.net/docs/installation.html)
    - Running the notebook also requires [jupyter](https://docs.jupyter.org/ru/latest/install/notebook-classic.html), [ipykernel](https://pypi.org/project/ipykernel/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html)


### 2. Set up a virtual environment with Python 3.6:
- Required for running the notebook for the Keyphrase extraction using EmbedRank in the `embedrank/`-directory

1. Create a venv with **Python 3.6** and make sure to activate it

2. Follow the installation instructions from the [EmbedRank repository](https://github.com/swisscom/ai-research-keyphrase-extraction)
    - Please note:
        - Installing sent2vec -> Don't do `git checkout f827d014a473aa22b2fef28d9e29211d50808d48` as instructed in the repo; this leads to an error because of a missing `setup.py`
        -  `pip install -r requirements.txt` might lead to some dependency issues; for me reinstalling some packages worked
            - **numpy 1.17.1**
            - **scikit_learn 0.19.0**
            - **scipy-0.19.1**
        - `nltk.download('punkt')` threw an error; please refer to this [thread](https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed)

3. Again, running the notebooks requires [jupyter](https://docs.jupyter.org/ru/latest/install/notebook-classic.html), [ipykernel](https://pypi.org/project/ipykernel/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/user_install.html)



## Author information
**Thao Van Liane Nguyen** (nguyen12@uni-potsdam.de)

