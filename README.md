# NLP News Text to Structured Database

## Project Overview

This project addresses the challenge of extracting structured information from unstructured news articles about road accidents in India. The primary goal is to create a robust system that can automatically extract and structure relevant information from news sources, addressing the lack of comprehensive accident databases in India.

## Project Architecture

The project is implemented in two main phases:
1. BERT-based Named Entity Recognition (NER)
2. GPT-3.5 Prompt Engineering

### Components

1. **Data Collection**
   - Web scraping of news articles
   - Article filtering system

2. **Text Processing Pipeline**
   - Clustering-based filtering using K-Means (TFIDF)
   - Similarity search using BERT embeddings
   - Exploratory Data Analysis (EDA)
     - Keyword location analysis within sentences
     - Validation of news article structure patterns

3. **NER Model Development**
   - Custom training dataset creation with JSON output format
   - Language model implementation:
     - Tokenization and vector database creation
     - Model selection: BERT, ROBERTA, XLNET
     - Spacy integration with preset NER parameters
   - Iterative testing with varying dataset sizes (200, 400, 600 samples)
   - Data augmentation using LLM
   - Performance metrics:
     - Accuracy scores > 80%
     - Entity transformation rules
     - ~40% information extraction rate

4. **GPT-3.5 Integration**
   - Prompt engineering and experimentation
   - High-level abstraction classes for testing (oneshot, multishot)
   - Custom evaluation metrics
   - JSON output structure optimization
   - Redundancy handling
   - Additional prompting layer for output classification

5. **Database and Visualization**
   - Structured database implementation
   - Custom querying system
   - Visualization dashboard development

## Technical Stack

- **NER Models**: BERT, ROBERTA, XLNET
- **Language Models**: GPT-3.5
- **Text Processing**: Spacy
- **Clustering**: K-Means with TFIDF
- **Embedding Models**: BERT-based similarity search
- **Output Format**: JSON

## Key Features

- Automated news article extraction
- Two-stage filtering process
- Custom NER model with high accuracy
- Prompt-engineered GPT-3.5 integration
- Structured database output
- Visualization capabilities

## Project Outcomes

- Successful extraction of structured information from unstructured news sources
- Creation of a queryable database of road accident information
- Development of visualization tools for data analysis
- Achievement of over 80% accuracy in entity extraction

## Future Improvements

- Enhanced data extraction rates
- Optimization of prompt engineering
- Expanded entity coverage
- Advanced visualization capabilities

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key for GPT-3.5 access
- Required Python packages:
  - transformers
  - torch
  - spacy
  - scikit-learn
  - pandas
  - numpy
  - beautifulsoup4 (for web scraping)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd NLPNewsTextToStructuredDB
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Download required Spacy model:
```bash
python -m spacy download en_core_web_lg
```

5. Set up environment variables:
```bash
export OPENAI_API_KEY='your-api-key'  # On Windows: set OPENAI_API_KEY=your-api-key
```

## Usage

### Input Data Format

Place your input data in the `06_prediction/input` directory:
- CSV file containing news articles (`filtered_dataset_oneliner.csv`)
- Pre-trained model data (`train_data.pickle`, `test_data.pickle`)
- Optional: ID list for annotation (`id_list_annotated.txt`)

### Making Predictions

1. Run the prediction notebook:
```bash
jupyter notebook 06_prediction/model_prediction_v1.ipynb
```

2. The model will process the input data and generate predictions in the `06_prediction/output` directory:
   - `predictions.csv`: Raw model predictions
   - `predictions_transform.csv`: Transformed predictions
   - `transformed.csv`: Final structured output

### Output Format

The final output (`transformed.csv`) contains structured information extracted from news articles including:
- Accident details
- Location information
- Vehicle information
- Casualty counts
- Temporal information
