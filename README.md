# Research Project: SSD Failure Effects and Performability with Machine Learning

## Table of Contents
1. [Objective](#objective)
2. [Description](#description)
3. [Background Information](#background-information)
4. [Process](#process)
5. [Repository Structure](#repository-structure)
6. [Notebooks](#notebooks)
   - [Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning](#comprehensive-analysis-of-ssd-failure-effects-and-performability-with-machine-learning)
   - [Data Processing and Exporting for HDD](#data-processing-and-exporting-for-hdd)
   - [MongoDB Population for HDD Data](#mongodb-population-for-hdd-data)
7. [Installation and Execution](#installation-and-execution)
8. [Scientific Paper](#scientific-paper)
9. [Datasets](#datasets)
   - [SSDs](#ssds)
   - [HDDs](#hdds)
10. [Funding](#funding)

## Objective
- Complete Exploratory Data Analysis.
- Predict hard drive failure using additional SMART statistics.

## Description
This repository contains Jupyter notebooks and a scientific paper on the analysis of SSD failures and performability using machine learning techniques. It also includes a comparative study of HDD data processing and exporting, and a guide on using MongoDB for storing and analyzing this data.

## Background Information
Data is an integral component of our society. From the simple caloric deficits collected in your Apple Watch to the user history in your Netflix account, data is used in a myriad of applications. With such an abundance of data being used daily, how is it stored? The solution is computer backup or cloud storage services. Furthermore, Backblaze is a world leader in computer backup and storage. Since 2013, Backblaze has published statistics and insights based on the hard drives in their data center. In this study, we’ll explore various features in a hard drive dataset to predict hard drive failure.

## Process
- Exploratory Data Analysis conducted utilizing various python packages (Numpy, Matplotlib, Pandas, and Plotly).
- Binary Classification Algorithms (Sci-Kit Learn):
  - Logistic Regression

# Datasets

## SSDs

### Solid-state drives (SSDs)
Directory: `ssd_open_data/`

This dataset includes nearly one million SSDs of 11 drive models from three vendors over a two-year span (January 2018 to December 2019). It is based on a dataset of SMART logs, trouble tickets, locations, and applications at Alibaba.

**Publication:**
"An In-Depth Study of Correlated Failures in Production SSD-Based Data Centers."  
Shujie Han, Patrick P. C. Lee, Fan Xu, Yi Liu, Cheng He, and Jiongzhou Liu.  
Proceedings of the 19th USENIX Conference on File and Storage Technologies (FAST 2021), February 2021.

### SMART logs of Solid-state drives (SSDs)
Directory: `ssd_smart_logs/`

This dataset includes nearly 500K SSDs of six drive models from three vendors over a two-year span. It contains daily SMART data and failures of six different SSD models ranging from January 1, 2018, to December 31, 2019.

**Publication:**
"General Feature Selection for Failure Prediction in Large-scale SSD Deployment."  
Fan Xu, Shujie Han, Patrick P. C. Lee, Yi Liu, Cheng He, and Jiongzhou Liu.  
Proceedings of the 51st IEEE/IFIP International Conference on Dependable Systems and Networks (DSN 2021), June 2021.

For each drive model, the serial number for an SSD in both datasets is identical, allowing for easy correlation of the two datasets by drive models and serial numbers.

**Link for the datasets:** [Alibaba SSD Datasets](https://github.com/alibaba-edu/dcbrain/blob/master/README.md)

## HDDs

This dataset contains information regarding HDDs from a Backblaze data center, including 231,309 HDDs from four manufacturers and 29 models. Backblaze has been monitoring these devices for eight years, during which 2,963 failures have occurred.

**Link for the datasets:** [Backblaze HDD Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data#downloadingTheRawTestData)


## Repository Structure
- `notebooks/`: Contains the Jupyter notebooks used in the analysis.
  - `Comprehensive_Analysis_of_SSD_Failure_Effects_and_Performability_with_Machine_Learning.ipynb`
  - `DataProcessingAndExportingHDD.ipynb`
  - `MongoDB_Population.ipynb`

## Notebooks

### Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning
This notebook provides a comprehensive analysis of SSD failures and performability using machine learning techniques.

**Content**:
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

**Content**:
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

**Content**:
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

## Installation and Execution

1. **Clone this repository**:
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

## Scientific Paper
The scientific paper produced with the results from the notebooks was published by Eric Borba, Reza Salkhordeh, Salim Mimouni, Eduardo Tavares, Paulo Maciel, Hossein Asadi, and André Brinkmann in the Proceedings of the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS), held in Potsdam, Germany, from May 14th to 16th, 2024.

- **Title**: A Hierarchical Modeling Approach for Assessing the Reliability and Performability of Burst Buffers
- **Abstract**: High availability is a crucial aspect of High-Performance Computing. Solid-state drives (SSD) offer peak bandwidth as node-local burst buffers. The limited write endurance of SSDs requires thorough investigation to ensure computational reliability. We propose a hierarchical model to evaluate the reliability and performability of burst buffers. We developed a machine-learning model to dynamically predict storage failures according to the wear caused by different applications. We also conducted an exploratory study to analyze the workload effects on SSD failures, and a representative dataset was adopted.
- **Presentation**:
The paper was presented by Eric Borba at the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS).

## Fundings

This work has been funded by the European Union’s Horizon 2020 JTI-EuroHPC research and innovation programme and the BMBF/DLR under the "IO-SEA" project with grant agreement number: 955811. This work was partially supported by Conselho Nacional de Desenvolvimento Científico e Tecnológico – CNPq under grant 202998/2019-3.