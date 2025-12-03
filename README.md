# HCMIU Data Mining Project

# Authors
Group O - HCMIU Data Mining Project

# Flims Mining - Anime Data Mining Project

## Overview
This project implements various machine learning classification and clustering algorithms for analyzing anime and user rating data. The implementation includes multiple classifiers such as Naive Bayes, OneR, J48 decision tree, ZeroR, K-Means clustering, and Apriori association rule mining algorithms using the Weka framework.

## Project Structure
```
.
├── dataset/
│   ├── raw/                      # Original raw datasets
│   │   ├── anime.csv            # Raw anime information
│   │   └── rating.csv           # Raw user ratings
│   ├── initial/                 # Initial preprocessing outputs
│   │   ├── anime-cleaned.arff
│   │   ├── rating-cleaned.arff
│   │   └── combined-cleaned.arff
│   └── advanced/                # Advanced preprocessing outputs
│       ├── anime-cleaned.arff
│       ├── rating-cleaned.arff
│       └── combined-cleaned.arff
├── pre-processing/              # Data preprocessing notebooks
│   ├── initial/
│   │   └── pre-processing-initial.ipynb
│   └── advanced/
│       └── pre-processing-advanced.ipynb
├── test-evaluation/             # Algorithm implementations
│   ├── initial/                 # Initial dataset implementations
│   │   ├── ZeroR_Classification.java
│   │   ├── OneR_Classification.java
│   │   ├── J48_Classification.java
│   │   ├── Naive_Bayes_Classification.java
│   │   ├── K_Means_Classification.java
│   │   └── Apriori_Classification.java
│   └── advanced/                # Advanced dataset implementations
│       ├── ZeroR_Classification.java
│       ├── OneR_Classification.java
│       ├── J48_Classification.java
│       ├── Naive_Bayes_Classification.java
│       ├── K_Means_Classification.java
│       └── Apriori_Classification.java
├── results/                     # Classification and clustering results
│   ├── initial/                 # Results from initial preprocessing
│   └── advanced/                # Results from advanced preprocessing
├── docs/                        # Project documentation
│   ├── DM_Movie_Mining_GroupO_Report.pdf
│   └── DM_Movie_Mining_GroupO_Presentation.pdf
├── lib/                         # External libraries and dependencies
└── algorithms/                  # Additional algorithm implementations
```

## Dataset Description
The project uses anime and user rating data:
- **anime.csv**: Contains anime information including titles, genres, types, episodes, ratings, and member counts
- **rating.csv**: Contains user ratings for various anime titles

### Data Files (ARFF Format)
The project uses ARFF (Attribute-Relation File Format) files for Weka processing:
- `anime-cleaned.arff`: Cleaned and processed anime data
- `rating-cleaned.arff`: Cleaned and processed rating data
- `combined-cleaned.arff`: Merged dataset combining anime and rating information

## Preprocessing Pipeline
The project includes two preprocessing approaches:

### 1. Initial Preprocessing (`pre-processing-initial.ipynb`)
- Basic data cleaning and transformation
- Simple encoding techniques
- Standard data merging
- Baseline preprocessing for comparison

### 2. Advanced Preprocessing (`pre-processing-advanced.ipynb`)
- Advanced data cleaning with sophisticated null handling
- Optimized encoding strategies
- Feature engineering
- Enhanced data quality controls
- Improved sampling techniques

## Implemented Algorithms

### Supervised Learning Classifiers

1. **ZeroR Classifier**
   - File: `ZeroR_Classification.java`
   - Baseline classifier that predicts the most frequent class
   - Used as a benchmark for other classifiers

2. **OneR Classifier**
   - File: `OneR_Classification.java`
   - Implementation of the OneR (One Rule) algorithm
   - Generates one rule for each predictor in the data

3. **J48 Decision Tree Classifier**
   - File: `J48_Classification.java`
   - Implementation of the C4.5 decision tree algorithm
   - Generates pruned or unpruned decision trees

4. **Naive Bayes Classifier**
   - File: `Naive_Bayes_Classification.java`
   - Probabilistic classifier based on Bayes' theorem
   - Assumes independence between features

### Unsupervised Learning Algorithms

5. **K-Means Clustering**
   - File: `K_Means_Classification.java`
   - Partitioning algorithm for cluster analysis
   - Groups similar data points together

6. **Apriori Association Rule Mining**
   - File: `Apriori_Classification.java`
   - Discovers frequent itemsets and association rules
   - Identifies patterns and relationships in the data

## Requirements
- **Java Development Kit (JDK)** 8 or higher
- **Weka Library** (included in `lib/` directory)
- **Python 3.x** (for preprocessing notebooks)
- **Jupyter Notebook** or **JupyterLab**
- **Python Libraries**:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Flims_Mining
```

### 2. Install Python Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Verify Weka Library
Ensure the Weka library is present in the `lib/` directory.

## Usage

### Data Preprocessing

1. **Run Initial Preprocessing**:
   ```bash
   jupyter notebook pre-processing/initial/pre-processing-initial.ipynb
   ```
   Execute all cells to generate cleaned datasets in `dataset/initial/`

2. **Run Advanced Preprocessing**:
   ```bash
   jupyter notebook pre-processing/advanced/pre-processing-advanced.ipynb
   ```
   Execute all cells to generate optimized datasets in `dataset/advanced/`

### Running Classification Algorithms

#### Compile Java Files
```bash
# For initial dataset
cd test-evaluation/initial
javac -cp "../../lib/*" *.java

# For advanced dataset
cd test-evaluation/advanced
javac -cp "../../lib/*" *.java
```

#### Run Individual Classifiers
```bash
# ZeroR Classifier
java -cp ".;../../lib/*" ZeroR_Classification

# OneR Classifier
java -cp ".;../../lib/*" OneR_Classification

# J48 Decision Tree
java -cp ".;../../lib/*" J48_Classification

# Naive Bayes
java -cp ".;../../lib/*" Naive_Bayes_Classification

# K-Means Clustering
java -cp ".;../../lib/*" K_Means_Classification

# Apriori Association Rules
java -cp ".;../../lib/*" Apriori_Classification
```

**Note**: On Linux/Mac, use `:` instead of `;` in the classpath:
```bash
java -cp ".:../../lib/*" ZeroR_Classification
```

## Results
Classification and clustering results are stored in the `results/` directory:
- `results/initial/`: Results from initial preprocessing pipeline
- `results/advanced/`: Results from advanced preprocessing pipeline

Each result file contains:
- Model evaluation metrics (accuracy, precision, recall, F-measure)
- Confusion matrix
- Detailed classification statistics
- Model-specific outputs (decision trees, rules, clusters, etc.)

## Project Documentation
Detailed project documentation is available in the `docs/` directory:
- **Report**: `DM_Movie_Mining_GroupO_Report.pdf`
- **Presentation**: `DM_Movie_Mining_GroupO_Presentation.pdf`

## License
This project is for educational purposes as part of the HCMIU Data Mining course.

---
*Last Updated: December 2025*
