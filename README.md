# Movie Reviews Visualization with Big Data Mining

## Project Overview
This project focuses on the visualization of movie reviews using data from the SST-2 (Stanford Sentiment Treebank) dataset. Text embeddings are generated using the **Universal Sentence Encoder (USE)**, and dimensionality reduction techniques like **t-SNE** are applied to analyze and visualize the data in a two-dimensional space.

## Table of Contents
- [Objective](#objective)
- [Assignment Description](#assignment-description)
- [Code Explanation](#code-explanation)
- [Data Visualization](#data-visualization)
- [Conclusions](#conclusions)
- [Requirements](#requirements)

## Objective
The goal of this project is to use the **Universal Sentence Encoder (USE)** to transform sentences into vectors and apply the **t-SNE** method to reduce the dimensionality of these vectors. The data is then visualized in a two-dimensional space, enabling analysis of movie reviews' sentiment.

## Assignment Description
1. **Loading and Preparing the SST-2 Dataset**: 
   - The SST-2 dataset, containing movie reviews categorized as positive or negative, is used.
2. **Text Vectorization**:
   - The movie reviews are transformed into vectors using the Universal Sentence Encoder (USE).
3. **Dimensionality Reduction with t-SNE**:
   - The t-SNE technique is applied to reduce the vector dimensionality to two components.
4. **Data Visualization**:
   - The reduced data is visualized in a scatter plot, where each point represents an embedded text, color-coded based on its label (positive or negative review).

## Code Explanation
- The code consists of the following key steps:
  1. **Importing Libraries**: Using `pandas`, `sklearn`, `tensorflow_hub`, and `matplotlib` for data handling, machine learning functions, and visualization.
  2. **Data Loading and Preparation**: The dataset is split into features (sentences) and labels, followed by a train-test split.
  3. **Text Vectorization with USE**: The Universal Sentence Encoder is used to generate vector embeddings for the text.
  4. **Dimensionality Reduction with t-SNE**: The high-dimensional vectors are reduced to two dimensions using t-SNE.
  5. **Visualization**: A scatter plot is created using `matplotlib` to visualize the reduced dimensions.

## Data Visualization
The visualization shows the distribution of movie review data in a two-dimensional space after the t-SNE method is applied. Each point represents an embedded text vector, and similar reviews are expected to cluster together. The points are color-coded based on the sentiment (positive or negative) of the review, providing insight into the distribution of sentiment in the dataset.

![screenshot](https://github.com/user-attachments/assets/5d36a9fb-b82a-4a7a-84ff-999e3563a418)


## Conclusions
- The **Universal Sentence Encoder (USE)** efficiently transforms each text into a high-dimensional numeric representation (embedding) that captures semantic information.
- The **t-SNE** method successfully reduces the high-dimensional embeddings to a two-dimensional space while preserving the local structure of the data, enabling the visualization of clusters and patterns.
- The scatter plot visualizes the embeddings in a two-dimensional space. Similar reviews tend to cluster together, and the color of each point corresponds to the review's sentiment.
- If clear clusters of reviews are observed in the plot, this suggests that reviews with similar sentiment are grouped together, indicating that the USE embeddings effectively capture semantic similarity.

## Requirements
* Python 3.x
* Required libraries:
  - pandas
  - sklearn
  - tensorflow_hub
  - matplotlib
