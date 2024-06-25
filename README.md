# Research Project: SSD Failure Effects and Performability with Machine Learning

## Description
This repository contains Jupyter notebooks and a scientific paper on the analysis of SSD failures and performability using machine learning techniques. It also includes a comparative study of HDD data processing and exporting, and a guide on using MongoDB for storing and analyzing this data.

## Repository Structure
- `notebooks/`: Contains the Jupyter notebooks used in the analysis.
  - `Comprehensive_Analysis_of_SSD_Failure_Effects_and_Performability_with_Machine_Learning.ipynb`
  - `DataProcessingAndExportingHDD.ipynb`
  - `MongoDB_Population.ipynb`

## Notebooks

### Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning
This notebook provides a comprehensive analysis of SSD failures and performability using machine learning techniques.

- **Content**:
  1. **Understanding SSD Failures**:
     - **Types of Failures**: Discussion on how SSDs fail differently from traditional HDDs.
     - **Common Causes**:
       - **Wear-Out**: Limited number of write/erase cycles.
       - **Power Loss**: Sudden power failures can cause data corruption.
       - **Firmware Bugs**: Issues within the drive’s firmware can lead to failures.
       - **Temperature Extremes**: Both high and low temperatures can affect SSD performance and longevity.
  2. **Performability Analysis**:
     - **Performability**: Combines performance and reliability of systems.
     - **Dynamic Mean Time To Failure (MTTF)**: A critical metric that evolves over time considering various operational conditions and usage patterns.
  3. **Machine Learning in Failure Prediction**:
     - **Predictive Models**: Machine learning algorithms can predict SSD failures by analyzing data patterns from various operational metrics.
     - **Features Used**: Temperature, read/write error rates, power cycles, etc.
     - **Algorithms**: 
       - **Random Forest Regressor**
       - **XGBoost Regressor**
       - **LSTM (Long Short-Term Memory) Networks**
       - **Voting Regressor**
     - **Data Collection and Processing**:
       - **Data Acquisition**: Gathering data from SSD sensors and logs.
       - **Preprocessing**: Cleaning and normalizing data for analysis.
       - **Feature Engineering**: Creating relevant features that capture the essence of potential failure points.
  4. **Results and Case Studies**:
     - Analysis of collected data shows significant patterns that can predict failures.
     - Case studies demonstrate the effectiveness of machine learning models in extending the life of SSDs and preventing unexpected downtimes.
     - **Datasets Used**: Various datasets collected from SSDs in different environments, detailing failure rates, operational conditions, and SMART attributes.
  5. **Proposed Models and Architecture**:
     - Description of the architecture used in the predictive models.
     - Comparison of different machine learning models and their performance metrics (e.g., Mean Absolute Error, Mean Squared Error, R2 Score).

### Data Processing and Exporting for HDD
This notebook addresses HDD data processing and export methods.

- **Content**:
  1. **HDD Data Characteristics**:
     - **Different Failure Mechanisms**: Comparison between HDD and SSD failures.
     - **Key Factors**:
       - **Mechanical Wear**: Moving parts are prone to wear and tear.
       - **Environmental Factors**: Vibration, temperature, and humidity can impact HDD reliability.
  2. **Data Processing Techniques**:
     - **Data Cleaning**: Removing noise and irrelevant data points.
     - **Normalization**: Scaling data to ensure consistency in analysis.
     - **Feature Selection**: Identifying important metrics like spindle speed, seek errors, and reallocated sectors.
  3. **Data Export Methods**:
     - **Efficient Data Export**: Methods that ensure seamless integration with analytical tools.
     - **Common Formats**: CSV, JSON, and database entries (e.g., MongoDB).
  4. **Tools and Frameworks**:
     - **Python Libraries**: Utilization of pandas for data manipulation.
     - **MongoDB**: Storing and querying large datasets efficiently.

### MongoDB Population for HDD Data
This notebook demonstrates setting up and populating HDD data into MongoDB.

- **Content**:
  1. **Setting Up MongoDB**:
     - **MongoDB NoSQL**: Suitable for handling large datasets.
     - **Steps**:
       - **Installation and Configuration**: Setting up MongoDB on a local or cloud server.
       - **Schema Design**: Planning the structure of database collections and documents.
  2. **Data Ingestion**:
     - **ETL (Extract, Transform, Load)**: Extracting data from various sources, transforming it into the required format, and loading it into MongoDB.
     - **Batch Processing**: Handling large volumes of data in batches to avoid performance bottlenecks.
  3. **Querying and Analysis**:
     - **Powerful Querying**: MongoDB’s querying capabilities to analyze stored data.
     - **Aggregation Framework**: Allows for complex data analysis and reporting.
  4. **Performance Optimization**:
     - **Indexing**: Creating indexes on important fields to speed up queries.
     - **Sharding**: Distributing data across multiple servers for better performance and reliability.

## Scientific Paper
The scientific paper produced with the results from the notebooks was published by Eric Borba, Reza Salkhordeh, Salim Mimouni, Eduardo Tavares, Paulo Maciel, Hossein Asadi, and André Brinkmann in the Proceedings of the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS), held in Potsdam, Germany, from May 14th to 16th, 2024.

- **Title**: A Hierarchical Modeling Approach for Assessing the Reliability and Performability of Burst Buffers
- **Abstract**: High availability is a crucial aspect of High-Performance Computing. Solid-state drives (SSD) offer peak bandwidth as node-local burst buffers. The limited write endurance of SSDs requires thorough investigation to ensure computational reliability. We propose a hierarchical model to evaluate the reliability and performability of burst buffers. We developed a machine-learning model to dynamically predict storage failures according to the wear caused by different applications. We also conducted an exploratory study to analyze the workload effects on SSD failures, and a representative dataset was adopted.
- **Funding**: This work has been funded by the European Union’s Horizon 2020 JTI-EuroHPC research and innovation programme and the BMBF/DLR under the "IO-SEA" project with grant agreement number: 955811. This work was partially supported by Conselho Nacional de Desenvolvimento Científico e Tecnológico – CNPq under grant 202998/2019-3.

## Installation and Execution

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/my-research-project.git
   cd my-research-project
   ```
2. Create a virtual environment and install the dependencies:
    ```python -m venv env
    source env/bin/activate  # or "env\Scripts\activate" on Windows
    pip install -r requirements.txt
    ```
3. Run the notebooks:
    ```
    jupyter notebook
    ```