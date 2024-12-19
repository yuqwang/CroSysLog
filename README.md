# CroSysLog

This project implements an AIOps tool CroSysLog for log-entry level anomaly detection (i.e., detect anomalous log entries) across different software systems.

## Repository Structure

- **`Learner.py`**: Defines the base model that do for log-event level anomaly detection in CroSysLog.
  
- **`Meta.py`**: Contains the implementation of the MAML algorithm. The `MAML` class manages the meta-training and meta-testing phase for CroSysLog. It trains the base model LSTM defined in `Learner.py`. We considered this repo **[MAML-Pytorch](https://github.com/dragen1860/MAML-Pytorch)** in our implementation.

- **`MetaDataset.py`**: Responsible for sampling, loading, pre-processing datasets for source and target systems. It defines the `MetaDataset` class, which samples the log data from source/target systems for meta-training and meta-testing phases, and create log embeddings using the neural representation method defined in `NeuralParser` class.

- **`NeuralParser.py`**: Implements the BERT-based embedding generation for logs using `BertTokenizer`(Wordpiece tokenization) and `BertModel` (base BERT). The `BertEmbeddings` class creates sentence embeddings for the logs, which are used as input to the meta-learning models.

- **`train_sample.py`**: Contains the training script for CroSysLog. This script uses the `MetaDataset` to load the data, and trains and evaluates CroSysLog. It also handles the training configuration and hyperparameter optimization using `ray[tune]`.

## Requirements

The project requires the following libraries:

- Python 3.x
- PyTorch
- Transformers (for BERT)
- Ray (for hyperparameter tuning)
- Pandas, NumPy, Polars (for data manipulation)
- Scikit-learn (for data normalization)

## Datasets

This project uses software log datasets from four large-scale distributed supercomputing systems—BGL, Thunderbird, Liberty, and Spirit—sourced from the [Usenix CFDR repository](https://www.usenix.org/cfdr). We do not hold the right to publicly share these datasets here. Please refer to the original source for downloading the datasets.

## How to Run

1. **Prepare the Dataset**
   
2. **Install Dependencies**

3. **Train CroSysLog**