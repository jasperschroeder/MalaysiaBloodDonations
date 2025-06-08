# MalaysiaBloodDonations
## Overview

This project analyzes a dataset of blood donations in Malaysia, aiming to uncover trends and build predictive models for daily donation counts. The repository includes data exploration, minor preprocessing and enhancement with holidays coming from the *holidays* package, running machine learning experiments and exposing the model's prediction endpoint via an API.

## Features

- Data cleaning and exploratory data analysis (EDA)
- Data validation using pydantic
- Visualizations of donation patterns
- Implementation of various neural network architectures
- Exposing a prediction endpoint of the model via an API

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/jasperschroeder/MalaysiaBloodDonations.git
    ```
2. Install dependencies:
    ```bash
    python -m venv .venv 
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```
3. Run the analysis notebooks one after the other.
4. To make use of the API calls shown in the notebook 3_API.ipynb, make sure to first run 
    ```bash
    uvicorn api:app --reload
    ```
to ensure that the API can receive request.


## Dataset

The dataset contains aggregated numbers of blood donations in Malaysia, split by blood type and state. The data can be found here: [Daily Blood Donations by Blood Group & State](https://data.gov.my/data-catalogue/blood_donations_state) and is made available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0). In my project I only aggregate this dataset and enrich it with holidays taken from the [holidays package](https://pypi.org/project/holidays/) which is made avilable under the [MIT License](https://spdx.org/licenses/MIT.html).


## License
This project is licensed under the MIT License.

