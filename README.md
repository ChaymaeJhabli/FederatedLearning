# DistilBERT Model for Depression Prediction

## Introduction

The project leverages DistilBERT, a streamlined variant of BERT, for predicting depression based on tweets. DistilBERT offers computational efficiency while maintaining strong performance, making it suitable for large-scale text analysis tasks.

## DistilBERT Model

DistilBERT employs a transformer architecture, similar to BERT, enabling it to understand bidirectional context in text sequences. Trained on a language task, DistilBERT learns intricate word representations and their contextual meanings, enabling nuanced understanding of text data.

### Training and Validation

During training on a dataset of annotated tweets for depression, the model demonstrates progressive improvement in accuracy. However, observations reveal a trend of overfitting from the second epoch onwards, indicating a need for regularization techniques or model adjustments.

### Results Obtained

Post-training evaluation showcases promising performance metrics. The model achieves a commendable accuracy of 99.17% on the training set and 91.81% on the validation set. Despite slight overfitting, the model showcases robust generalization capabilities, evidenced by an accuracy of 92.67% on the test dataset.

## Conclusion

In conclusion, the DistilBERT model presents a viable approach for depression prediction from tweets, offering satisfactory accuracy levels. The observed overfitting warrants further exploration into regularization techniques or model fine-tuning. Overall, the model demonstrates promising potential for real-world applications in mental health monitoring and analysis.


# Federated Learning for DistilBERT

## Overview
This project explores the application of federated learning techniques to train a DistilBERT model for sentiment analysis(Our case treat depression prediction.). Federated learning allows training a global model across decentralized devices or servers holding local data samples without exchanging them. In this project, we train the model on data partitions held by multiple clients, mimicking real-world scenarios where data privacy and distribution concerns arise.

## Methodology
### Data Preprocessing
- Data cleaning: Tweets were preprocessed by removing mentions, hashtags, URLs, and special characters. Text was also converted to lowercase.
- Data partitioning: Two methods were employed:
  - IID (Independent and Identically Distributed) partitioning: Data was randomly split among clients, ensuring each client's data distribution resembled the global dataset.
  - Non-IID partitioning: Data was sorted based on labels, and shards were distributed among clients, introducing heterogeneity in client datasets.

### Model Architecture
- Customized DistilBERT model: A DistilBERT model was fine-tuned for sentiment analysis. A dropout layer and a dense layer were added on top of the DistilBERT base to produce the final output.

### Training
- Training rounds: The training process involved multiple rounds where clients locally trained their models on their data partitions and then communicated the updates to a central server.
- Training parameters: 
  - Learning rate: 1e-05
  - Batch size: 10
  - Number of clients: 10
  - Number of training passes on local datasets for each round: 1
  - Patience for early stopping: 3 epochs
- Loss function: Cross-entropy loss

### Testing
- Test dataset: Validation dataset was used for testing.
- Test methodology: Model performance was evaluated using a separate validation dataset, leveraging PyTorch DataLoader.

## Results
### IID Partitioning
- **Training Accuracy:** Achieved an average accuracy of 88.74% over multiple rounds. In IID partitioning, data is randomly distributed among clients, ensuring that each client's dataset is representative of the overall data distribution. This randomness introduces diversity in the training process, leading to slightly lower training accuracies compared to non-IID partitioning.
- **Validation Accuracy:** Reached up to 92.78%. The validation accuracy reflects the model's ability to generalize to unseen data, demonstrating its effectiveness in capturing patterns present in the validation dataset.
- **Test Accuracy:** Approximately 91.12%. The test accuracy provides a measure of the model's performance on an independent dataset, further validating its effectiveness in real-world scenarios.

### Non-IID Partitioning
- **Training Accuracy:** Attained an average accuracy of 99.67% over multiple rounds. Non-IID partitioning sorts the data based on labels, resulting in each client holding a specific subset of labels. This structured distribution can lead to overfitting, as the model may become highly specialized in predicting the labels present in the local datasets. Consequently, the training accuracy tends to be significantly higher compared to IID partitioning.
- **Validation Accuracy:** Fluctuated around 80.84%. The fluctuation in validation accuracy suggests that the model's performance varies when evaluated on different subsets of the validation dataset. This variability may stem from the limited diversity in the training data, which affects the model's ability to generalize.
- **Test Accuracy:** Approximately 80.84%. The test accuracy reflects the model's performance on unseen data, highlighting its capability to make predictions in real-world scenarios. The lower test accuracy compared to IID partitioning indicates that the model may struggle to generalize to new data distributions, potentially due to overfitting on the training data.

### Explanation
- **Difference Between IID and Non-IID:** The main difference between IID and non-IID partitioning lies in the distribution of data among clients. IID partitioning ensures randomness in data distribution, promoting diversity in training and generalization to new data. In contrast, non-IID partitioning leads to structured distributions, which may result in overfitting and limited generalization.
- **Overfitting in Non-IID Partitioning:** The high training accuracy observed in non-IID partitioning suggests overfitting, where the model memorizes patterns specific to the training data without effectively capturing underlying trends in the overall dataset. This overfitting phenomenon is more pronounced in non-IID partitioning due to the structured nature of the data distribution, leading to reduced performance on unseen data.

In the context of federated learning, understanding the impact of data partitioning schemes is crucial for optimizing model performance and ensuring robustness across diverse datasets. By evaluating the model's performance under different partitioning strategies, we gain insights into its ability to generalize and make accurate predictions in real-world scenarios.

## Conclusion
- Federated learning with IID partitioning showed promising results, achieving competitive accuracies on both training and validation datasets.
- Non-IID partitioning, while resulting in high training accuracies, exhibited challenges in generalizing to unseen data, as evidenced by lower validation and test accuracies.
- Further research and experimentation are warranted to optimize federated learning approaches for DistilBERT models, especially in scenarios involving non-IID data distributions.
