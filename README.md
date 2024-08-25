
![titanic_Kaggle](https://github.com/user-attachments/assets/6c861b1c-b86b-4b92-b7d3-792d504f8bd7)

# Titanic Survival Prediction
 with Logistic Regression

This repository contains a data analysis project focused on predicting the survival of passengers from the Titanic disaster using a logistic regression model. The analysis utilizes Python and various data science libraries to explore, clean, and model the Titanic dataset.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Data Preparation](#data-preparation)
6. [Feature Engineering](#feature-engineering)
7. [Modeling](#modeling)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Contributing](#contributing)


## Project Overview

The goal of this project is to apply logistic regression to predict whether a passenger on the Titanic survived based on various features such as age, sex, class, and other characteristics. The project demonstrates essential data science techniques including data cleaning, feature engineering, and model evaluation.

## Dataset

The dataset used in this analysis is from the famous Titanic competition on [Kaggle](https://www.kaggle.com/c/titanic). It contains information on passengers who were aboard the Titanic when it sank on April 15, 1912.

### Dataset Files

- `train.csv`: Training dataset containing passenger information and survival status.
- `test.csv`: Testing dataset containing passenger information without survival status.

## Installation

To run this project, you need Python installed along with the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Project Structure

The repository is organized as follows:

```
.
├── data
│   ├── train.csv
│   └── test.csv
├── notebooks
│   └── Titanic Basic Solution with Logistic Regression-2.ipynb
├── src
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   └── modeling.py
├── README.md
```

- `data/`: Contains the dataset files.
- `notebooks/`: Contains Jupyter notebooks used for exploratory data analysis and model building.
- `src/`: Contains Python scripts for data preparation, feature engineering, and modeling.
- `README.md`: Project documentation.
- `LICENSE`: License for the repository.

## Data Preparation

The data preparation process involves loading the datasets, handling missing values, and merging the training and test sets for uniform preprocessing.

### Steps:

1. **Load Data**: The `train.csv` and `test.csv` files are loaded using `pandas`.
2. **Explore Data**: An initial exploration is performed to understand the data structure and identify missing values.
3. **Handle Missing Values**: Missing values in columns such as `Age`, `Fare`, and `Embarked` are filled using appropriate strategies like median or mode.

## Feature Engineering

Feature engineering involves creating new features or modifying existing ones to improve model performance.

### Created Features:

- **Age Binning**: Age is grouped into categories (0: <=16, 1: 17-32, 2: 33-48, 3: 49-64, 4: >64).
- **Title Extraction**: Titles such as Mr., Mrs., Miss., etc., are extracted from passenger names.
- **Cabin**: Cabin information is simplified to its first letter.
- **Family Size**: Calculated as the sum of `SibSp` (siblings/spouses) and `Parch` (parents/children) plus one.
- **IsAlone**: Binary feature indicating if a passenger is alone (no family aboard).

## Modeling

The modeling process involves training a logistic regression model on the processed dataset.

### Steps:

1. **Train/Test Split**: The training data is split into training and validation sets.
2. **Model Training**: A logistic regression model is trained on the training set.
3. **Evaluation**: The model is evaluated using accuracy, precision, recall, and F1 score.

## Results

The logistic regression model achieves satisfactory results on the validation set, indicating its effectiveness in predicting survival based on the selected features.

## Conclusion

This project demonstrates the process of building a predictive model using logistic regression for the Titanic dataset. It covers data cleaning, feature engineering, and model evaluation, providing a comprehensive overview of a typical data science workflow.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or additions.


---

Feel free to customize the README further to match any specific details or preferences for your GitHub repository.
![image](https://github.com/user-attachments/assets/14ba6e85-f02f-422d-9ea6-c2feda9da28a)
