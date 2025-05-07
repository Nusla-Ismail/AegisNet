# Fraud Detection using Graph Attention Networks (GAT)

This project implements a fraud detection system using Graph Attention Networks (GAT) to identify fraudulent transactions. Unlike traditional approaches that treat each transaction independently, this model represents the data as a graph to capture relationships and temporal patterns.

## Table of Contents

- [Features](#features)
- [Model Overview](#model-overview)
- [Benchmarking](#benchmarking)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Visualization](#visualization)
- [Technology Stack](#technology-stack)
- [Security](#security)
- [Future Work](#future-work)

## Features

- Graph-based modeling of transaction data
- Temporal dynamic graph construction
- Multi-head Graph Attention Network (GAT) layers
- Basic user login/signup interface using Flask
- Upload and analysis of transaction CSV files
- Fraud insights and visual analytics
- Benchmarking with traditional ML models

## Model Overview

- **Graph Construction**: Nodes represent transactions. Edges are created using K-Nearest Neighbors (KNN) based on feature embeddings.
- **Dynamic Graph Updates**: The graph structure is updated every 10 epochs using the latest node embeddings.
- **Model Architecture**: Three-layer GAT with multi-head attention to learn node representations.
- **Classification**: Binary classification using Binary Cross-Entropy Loss to predict fraud.

## Benchmarking

The GAT-based model is compared with XGBoost, Random Forest, and Linear Regression.

| Model             | Precision | Recall | F1-Score |
| ----------------- | --------- | ------ | -------- |
| GAT-DGNN          | Moderate  | High   | Good     |
| XGBoost           | High      | Lower  | Higher   |
| Random Forest     | Good      | Medium | Good     |
| Linear Regression | Low       | Low    | Low      |

The GAT model excels in recall, which is critical in fraud detection for identifying actual fraudulent transactions.

## Installation

```bash
git clone https://github.com/Nusla-Ismail/AegisNet.git
cd AegisNet
pip install -r requirements.txt
python app.py
```
