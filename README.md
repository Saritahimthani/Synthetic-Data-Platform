# Synthetic Data Generation Platform with Privacy Guarantees

This project implements a platform for generating synthetic data that preserves the statistical properties of the original data while providing privacy guarantees.

## Directory Structure

```
synthetic-data-platform/
├── data/                      # Data storage
│   ├── raw/                   # Original datasets
│   ├── processed/             # Preprocessed datasets
│   └── synthetic/             # Generated synthetic datasets
├── models/                    # Model implementations
│   ├── gan/                   # GAN-based models
│   ├── dp/                    # Differential privacy models
│   └── hybrid/                # Combined approaches
├── evaluation/                # Evaluation metrics and tools
│   ├── utility/               # Statistical similarity metrics
│   ├── privacy/               # Privacy attack simulations
│   └── reports/               # Generated evaluation reports
├── preprocessing/             # Data preprocessing pipelines
├── visualization/             # Visualization tools
├── app/                       # Web interface
├── notebooks/                 # Jupyter notebooks for experiments
├── tests/                     # Unit and integration tests
├── config/                    # Configuration files
└── docs/                      # Documentation
```

## Project Overview

This guide will walk you through building a comprehensive synthetic data generation platform that preserves privacy while maintaining the utility of the original data. The platform will support tabular data (the most common use case) and implement both GAN-based and differential privacy approaches.

## Project Contributers

Sarita H
Arya K