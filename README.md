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

### MongoDB Population for SSD and HDD Data

This notebook demonstrates setting up MongoDB and populating SSD and HDD data into MongoDB.

**Content**:
1. **Setting Up MongoDB**:
   - **MongoDB NoSQL**: MongoDB is chosen for its ability to handle large datasets efficiently.
   - **Steps**:
     - **Installation and Configuration**: MongoDB can be set up on a local or cloud server. Ensure MongoDB is installed and configured properly.
     - **Schema Design**: Plan the structure of database collections and documents to accommodate SSD and HDD data effectively.

2. **Data Ingestion**:
   - **ETL (Extract, Transform, Load)**: Data is extracted from various sources, transformed as needed, and loaded into MongoDB collections.
   - **Batch Processing**: Handle large volumes of SSD and HDD data in batches to optimize performance during ingestion.

3. **MongoDB Operations and Data Management**:
   - **Database Creation and Collection Management**:
     - **Objective**: Create MongoDB database and manage collections.
     - **Approach**: Use `use mydb` to create and switch to a database named `mydb`. Create collections using `db.createCollection`. View existing collections with `show collections`.
   
   - **Data Loading Using mongoimport**:
     - **Objective**: Load data into MongoDB from a JSON file.
     - **Approach**: Utilize `mongoimport` command-line tool to import SSD and HDD data into respective collections.
   
   - **Basic Data Querying**:
     - **Objective**: Perform basic queries to retrieve SSD and HDD data from MongoDB collections.
     - **Approach**: Use `db.collection.find()` to query documents based on specified criteria.

   - **Counting Documents in a Collection**:
     - **Objective**: Count the number of documents in a MongoDB collection.
     - **Approach**: Implement `db.collection.count()` to get the document count in a collection.

   - **Database Administration**:
     - **Objective**: Perform basic administrative tasks for MongoDB databases.
     - **Approach**: Use `db.dropDatabase()` to delete the current database (`mydb`).

4. **Querying and Analysis**:
   - **Powerful Querying**: Leverage MongoDB’s querying capabilities to analyze stored SSD and HDD data.
   - **Aggregation Framework**: Utilize MongoDB's aggregation framework for complex data analysis and reporting.

5. **Performance Optimization**:
   - **Indexing**: Create indexes on important fields (e.g., serial numbers, timestamps) in SSD and HDD data collections to improve query performance.
   - **Sharding**: Consider sharding MongoDB collections to distribute data across multiple servers for scalability and enhanced performance.

### Summary

This section outlines the setup of MongoDB for handling SSD and HDD data, including data population, basic operations, querying, and performance optimization strategies. These steps are crucial for effectively managing and analyzing large datasets in MongoDB.

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

This notebook provides a detailed exploration of HDD failures across various dimensions, analyzing key metrics such as reallocated sectors, uncorrectable errors, command timeouts, and more. The analysis aims to understand the relationship between these metrics and the number of written blocks, and to calculate performability metrics such as AFR (Annualized Failure Rate) and MTTF (Mean Time To Failure).

**Content**:
1. **Investigation of Reallocated Sectors**
    - **Objective**: Analyze the relationship between the number of blocks written and reallocated sectors.
    - **Approach**:
        - Group data by day to calculate the daily mean of blocks written and reallocated sectors.
        - Plot reallocated sectors against blocks written to visualize the trend.

2. **Investigation of Uncorrectable Errors**
    - **Objective**: Examine the correlation between blocks written and uncorrectable errors.
    - **Approach**:
        - Group data by day to compute the daily mean of blocks written and uncorrectable errors.
        - Plot uncorrectable errors against blocks written for analysis.

3. **Investigation of Command Timeouts**
    - **Objective**: Assess how command timeouts are affected by the number of blocks written.
    - **Approach**:
        - Group data by day to get the daily mean of blocks written and command timeouts.
        - Remove outliers to ensure data integrity.
        - Plot command timeouts against blocks written.

4. **Investigation of Current Pending Sector Count**
    - **Objective**: Determine the impact of written blocks on the current pending sector count.
    - **Approach**:
        - Group data by day to calculate the daily mean of blocks written and current pending sector count.
        - Plot the current pending sector count against blocks written.

5. **Investigation of Uncorrectable Sector Count**
    - **Objective**: Analyze the effect of written blocks on the uncorrectable sector count.
    - **Approach**:
        - Group data by day to compute the daily mean of blocks written and uncorrectable sector count.
        - Plot uncorrectable sector count against blocks written.

6. **Analysis of HDD Failures by Blocks Written**
    - **Objective**: Correlate the number of HDD failures with the number of blocks written.
    - **Approach**:
        - Load failure data for specific HDD models.
        - Group data by day and calculate cumulative sums of disk IDs and blocks written.
        - Plot the number of HDD failures against blocks written.

7. **Correlation Analysis Between Failures and Written Blocks**
    - **Objective**: Compute the correlation between the number of failures and the blocks written.
    - **Approach**:
        - Calculate cumulative sums of operational days and blocks written.
        - Compute the correlation between these metrics for all HDD models.

**Summary**:
This exploratory analysis offers comprehensive insights into HDD failures, focusing on various metrics and their relationship with the number of blocks written. By understanding these factors, we can improve HDD reliability and optimize their usage in different environments.

### Exploratory Analysis SSD
This notebook provides an in-depth exploration of SSD failures across various applications, analyzing failure rates and performability metrics such as AFR (Annualized Failure Rate) and MTTF (Mean Time To Failure).

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

**Summary**:
This exploratory analysis provides insights into SSD failures across various dimensions including application, model, flash technology, capacity, and lithography. By understanding these factors, we can improve SSD reliability and optimize their usage in different environments.

### Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning

This notebook provides a comprehensive exploration of SSD failures and performability metrics across various applications, leveraging machine learning models to enhance the analysis. The objective is to understand the factors affecting SSD reliability and improve their usage in different environments.

**Content**:
1. **Data Loading and Preprocessing**
   - **Objective**: Load and preprocess the SSD datasets for analysis.
   - **Approach**:
     - Read the data from CSV files.
     - Handle missing values and clean the data for further analysis.
     - Encode categorical variables and normalize numerical features.

2. **Exploratory Data Analysis (EDA)**
   - **Objective**: Explore the dataset to understand the distribution and relationships of variables.
   - **Approach**:
     - Generate summary statistics and visualizations for key variables.
     - Identify correlations and patterns in the data.
     - Analyze the distribution of SSD failures across different applications.

3. **Investigation of SSDs by Application**
   - **Objective**: Analyze SSD failure rates and performability metrics based on the application in which they are used.
   - **Approach**:
     - Identify unique applications from the dataset.
     - Calculate total SSDs, failed SSDs, AFR (Annualized Failure Rate), and MTTF (Mean Time To Failure) for each application.
     - Differentiate between general SSDs and common SSDs across multiple datasets.

4. **Investigation of AFR/MTTF and wear-related SMART attributes per SSD Model for Each Application**
   - **Objective**: Assess the reliability of different SSD models within each application.
   - **Approach**:
     - Merge datasets to associate SSD models with failure data.
     - Loop through each application and SSD model to calculate AFR and MTTF.
     - Compile statistics to track the evolution of failures in SSDs and wear-related smart attributes (such as reallocated sectors count and wear leveling) in relation to the number of written blocks per application.

5. **Machine Learning Model Development**
   - **Objective**: Develop and validate machine learning models to predict SSD failures.
   - **Approach**:
     - Split the data into training and testing sets.
     - Train various machine learning models, including Random Forest, LSTM, and XGBoost.
     - Perform hyperparameter tuning to optimize model performance.
     - Evaluate models using metrics such as accuracy, precision, recall, and F1-score.
     - Select the best-performing model for further analysis.

6. **Model Validation and Performance Evaluation**
   - **Objective**: Validate the performance of the selected machine learning model.
   - **Approach**:
     - Perform cross-validation to ensure model robustness.
     - Generate confusion matrices and ROC curves to assess model performance.
     - Interpret model predictions and identify key features influencing SSD failures.

7. **Performability Analysis**
   - **Objective**: Analyze the performability of SSDs, considering both performance and reliability.
   - **Approach**:
     - Calculate metrics such reliability and performance (when impacted by failures).
     - Compare performability metrics across different applications.
     - Identify factors affecting SSD performability and suggest improvements.
     - Optimization using the composite desirability statistical technique, considering performability, MTTF, and MTTR metrics, for a scenario involving LQCD and ECMWF applications, the Gekko file system, and 512 burst buffers.
     
**Summary**:
This comprehensive analysis provides critical insights into SSD failures and performability across various dimensions. By leveraging machine learning models and performability analysis, we can enhance the reliability and performance of SSDs in different environments, ultimately improving the overall effectiveness of Burst Buffer systems.

### Utils

This notebook provides a detailed analysis of system reliability metrics using Reliability Block Diagram (RBD) analysis and approximation of delay distributions for GekkoFS using phase-type distributions.

**Content**:

1. **Computing System Reliability Metrics using RBD Analysis**
    - **Objective**: Calculate system Mean Time Between Failures (MTBF) and Mean Time To Repair (MTTR) metrics for different applications using RBD analysis in either series or parallel configuration.
    - **Approach**:
        - Define MTBF and MTTR values for different applications and the number of blocks in the system.
        - For each application and configuration, compute the system MTBF and MTTR based on series or parallel configuration.
        - Returns a DataFrame with application names, number of blocks, input MTBF and MTTR, and computed system metrics.

2. **Approximating GekkoFS Delay Distributions**
    - **Objective**: Approximate GekkoFS delay distribution using phase-type distributions based on moment matching technique.
    - **Approach**:
        - Calculate the relationship between mean (mu_d) and standard deviation (sigma_d) of the delay.
        - Determine the appropriate phase-type distribution (single timing transition, Erlang, hypoexponential, or hyperexponential) based on this relationship.
        - Returns a description of the phase-type distribution approximation or "Unknown distribution" if the distribution cannot be determined.

3. **CSV Data Processing for Phase-Type Approximation**
    - **Objective**: Process GekkoFS performance data from CSV files, calculate phase-type distributions and their parameters, and save the results into a CSV file.
    - **Approach**:
        - Load average write data and standard deviation data from CSV files.
        - For each node and object size, calculate sigma_d (absolute standard deviation) and apply the phase-type approximation function.
        - Extract and store parameters of the phase-type distribution.
        - Create a DataFrame with the results and save it to a new CSV file.

**Summary**:
This notebook conducts detailed calculations of system reliability metrics using RBD analysis and approximates GekkoFS delay distributions using phase-type distributions based on moment matching. These methods provide deep insights into system reliability metrics and GekkoFS delay patterns, offering valuable insights to enhance system reliability and performance.

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