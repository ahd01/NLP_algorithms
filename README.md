# NLP_algorithms
Table of Contents

    1-Project Overview
    2-Dataset
    3-Models Used
    4-Requirements
    5-Installation
    6-How to Run
    7-Evaluation
    8-Example Sentences for Testing
    9-Results Visualization
Project Overview

The goal of this project is to classify the sentiment of Arabic text (positive/negative) by leveraging pre-trained BERT embeddings and different RNN-based models for downstream classification. The models are trained on an Arabic sentiment dataset using a variety of architectures to compare performance.
Dataset

The dataset used in this project is the 330k Arabic Sentiment Reviews dataset available on Kaggle. It consists of 330,000 reviews labeled as either positive or negative. The dataset was downloaded using the kagglehub library.

    Dataset: 330k Arabic Sentiment Reviews

Models Used

We trained and evaluated the following models using BERT embeddings:

    Simple RNN
    LSTM (Long Short-Term Memory)
    Bidirectional RNN
    Bidirectional LSTM

These models are trained on BERT embeddings extracted from the aubmindlab/bert-base-arabertv02 model.
Requirements

Before running the project, ensure you have the following libraries installed:

    Python 3.x
    PyTorch
    TensorFlow/Keras
    HuggingFace Transformers
    Pandas, NumPy, Matplotlib, scikit-learn
    kagglehub (for dataset downloading)
    To install the dependencies, you can run:

pip install torch tensorflow transformers pandas numpy scikit-learn matplotlib wandb kagglehub tqdm

Installation

    Clone the repository to your local machine:

https://github.com/ahd01/NLP_algorithms.git
cd your-repo-folder

Install the required libraries:

    pip install -r requirements.txt

    Set up your Kaggle API key to download the dataset using kagglehub (if you haven't already). Follow the instructions here.

How to Run

    Download the Dataset:

    The dataset will automatically download when you run the code:

path = kagglehub.dataset_download("abdallaellaithy/330k-arabic-sentiment-reviews")

Preprocess Data:

The BERT tokenizer and model will preprocess the dataset into embeddings. You don't need to manually preprocess the text.

Train the Models:

You can run the models by executing the Python code. Example:

# Train the Simple RNN model
rnn_model.fit(train_embeddings, y_train, epochs=5, batch_size=32, validation_data=(test_embeddings, y_test))

# Evaluate the Simple RNN model
rnn_model_loss, rnn_model_accuracy = rnn_model.evaluate(test_embeddings, y_test, verbose=2)

Test on Custom Sentences:

You can pass any sentence to the model for testing:

    sentence = "أحببت هذا الفيلم كثيرًا، كانت تجربة رائعة!"
    predictions = test_on_sentence(sentence)
    print(predictions)

Evaluation

For evaluation, we use accuracy, precision, recall, and F1 score. These metrics are computed for each model using the test data.

You can visualize the comparison of these metrics with bar charts:

evaluate_and_plot()

Example Sentences for Testing

Here are some example sentences for testing different models:

    Positive Sentiment:
        "أحببت هذا الفيلم كثيرًا، كانت تجربة رائعة!"
        "الخدمة في هذا المطعم كانت ممتازة والطعام لذيذ جدًا!"

    Negative Sentiment:
        "لم تعجبني جودة المنتج، كان مخيبًا للآمال."
        "التجربة كانت سيئة جدًا، لا أوصي بهذا المكان أبدًا."

    Neutral Sentiment:
        "كان الحفل عاديًا ولم يكن هناك شيء مميز."
        "المنتج يؤدي الغرض المطلوب، ولكنه ليس استثنائيًا."

Results Visualization

After evaluating the models, you can generate visual comparisons (accuracy, precision, recall, F1 score) by running:

evaluate_and_plot()

This will generate bar charts comparing each model's performance across the different metrics.
