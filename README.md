# Research Project: SSD Failure Effects and Performability with Machine Learning

## Table of Contents
1. [Objective](#objective)
2. [Description](#description)
3. [Background Information](#background-information)
4. [Process](#process)
5. [Repository Structure](#repository-structure)
6. [Notebooks](#notebooks)
   - [MongoDB Population for SSD and HDD data](#mongodb-population)
   - [Data Processing and Exporting for HDD](#data-processing-and-exporting-for-hdd)   
   - [Exploratory Analysis HDD](#exploratory-analysis-hdd)
   - [Exploratory Analysis SSD](#exploratory-analysis-ssd)
   - [Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning](#comprehensive-analysis-of-ssd-failure-effects-and-performability-with-machine-learning)
   - [Utils](#utils)
7. [Installation and Execution](#installation-and-execution)
8. [Publication](#scientific-paper)
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
  - `ExploratoryAnalysisHDD.ipynb`
  - `ExploratoryAnalysisSSD.ipynb`
  - `utils.ipynb`

## Notebooks

### MongoDB Population for SSD and HDD data
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

### Exploratory Analysis HDD

### Exploratory Analysis SSD
This notebook provides an in-depth exploration of SSD failures across various applications, analyzing failure rates and performability metrics such as AFR (Annualized Failure Rate) and MTTF (Mean Time To Failure).

**Summary**:
This exploratory analysis provides insights into SSD failures across various dimensions including application, model, flash technology, capacity, and lithography. By understanding these factors, we can improve SSD reliability and optimize their usage in different environments.

**Content**:
1. **Investigation of SSDs by Application**
- **Objective**: Analyze SSD failure rates and performability metrics based on the application in which they are used.
- **Approach**:
  - Identify unique applications from the dataset.
  - Calculate total SSDs, failed SSDs, AFR, and MTTF for each application.
  - Differentiate between general SSDs and common SSDs across multiple datasets.

2. **Investigation of SSDs Without Application Distinction**
- **Objective**: Calculate overall failure rates and performability metrics without distinguishing between applications.
- **Approach**:
  - Aggregate data to compute general AFR and MTTF.
  - Compare general failure statistics with application-specific statistics.

3. **Investigation of AFR and MTTF per SSD Model for Each Application**
- **Objective**: Assess the reliability of different SSD models within each application.
- **Approach**:
  - Merge datasets to associate SSD models with failure data.
  - Loop through each application and SSD model to calculate AFR and MTTF.
  - Compile statistics to identify models with higher failure rates.

4. **Investigation of AFR and MTTF per SSD Flash Technology for Each Application**
- **Objective**: Determine the impact of different flash technologies (e.g., MLC, 3D-TLC) on SSD reliability.
- **Approach**:
  - Classify SSD models based on flash technology.
  - Group data by application and flash technology to calculate failure metrics.
  - Compare AFR and MTTF across different technologies.

5. **Investigation of AFR and MTTF per SSD Capacity for Each Application**
- **Objective**: Analyze how SSD capacity affects failure rates and reliability.
- **Approach**:
  - Classify SSDs by capacity (e.g., 480GB, 1920GB).
  - Group data by application and capacity to compute failure metrics.
  - Assess the correlation between SSD capacity and failure rates.

6. **Investigation of AFR and MTTF per SSD Lithography for Each Application**
- **Objective**: Evaluate the reliability of SSDs based on their lithography.
- **Approach**:
  - Classify SSDs by lithography node (e.g., 20nm, 16nm).
  - Group data by application and lithography to calculate AFR and MTTF.
  - Identify lithographies with better reliability profiles.

7. **Analysis of SSD Blocks Written/Read per Application**
- **Objective**: Examine the relationship between data written/read and SSD reliability.
- **Approach**:
  - Merge datasets to include SMART attributes and SSD usage statistics.
  - Calculate total number of blocks written/read and median wearout indicators per application.
  - Determine the percentage of data written versus read and correlate with failure metrics.

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

### Utils

## Publication
A scientific paper based on the results from the notebooks was authored by Eric Borba, Reza Salkhordeh, Salim Mimouni, Eduardo Tavares, Paulo Maciel, Hossein Asadi, and André Brinkmann. It was published in the Proceedings of the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS), held in Potsdam, Germany, from May 14th to 16th, 2024.

- **Title**: A Hierarchical Modeling Approach for Assessing the Reliability and Performability of Burst Buffers
- **Abstract**: High availability is a crucial aspect of High-Performance Computing. Solid-state drives (SSD) offer peak bandwidth as node-local burst buffers. The limited write endurance of SSDs requires thorough investigation to ensure computational reliability. We propose a hierarchical model to evaluate the reliability and performability of burst buffers. We developed a machine-learning model to dynamically predict storage failures according to the wear caused by different applications. We also conducted an exploratory study to analyze the workload effects on SSD failures, and a representative dataset was adopted.
- **Presentation**:
The paper was presented by Eric Borba at the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS).

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

## Fundings

This work has been funded by the European Union’s Horizon 2020 JTI-EuroHPC research and innovation programme and the BMBF/DLR under the "IO-SEA" project with grant agreement number: 955811. This work was partially supported by Conselho Nacional de Desenvolvimento Científico e Tecnológico – CNPq under grant 202998/2019-3.