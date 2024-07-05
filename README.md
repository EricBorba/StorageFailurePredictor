# Investigating SSD Reliability and Performability in High-Performance Computing: Analytical Modeling, Machine Learning, and Exploratory Data Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Methodology](#methodology)
4. [Datasets](#datasets)
   - [SSDs](#ssds)
   - [HDDs](#hdds)
   - [Real Traces](#real-traces)
5. [Repository Structure](#repository-structure)
6. [Notebooks](#notebooks)
   - [MongoDB Population for SSD and HDD Data](#mongodb-population-for-ssd-and-hdd-data)
   - [Data Processing and Exporting for HDD](#data-processing-and-exporting-for-hdd)
   - [Exploratory Analysis HDD](#exploratory-analysis-hdd)
   - [Exploratory Analysis SSD](#exploratory-analysis-ssd)
   - [Comprehensive Analysis of SSD Failure Effects and Performability with Machine Learning](#comprehensive-analysis-of-ssd-failure-effects-and-performability-with-machine-learning)
   - [Utils](#utils)
7. [Installation and Execution](#installation-and-execution)
8. [Publication](#publication)
9. [Funding](#funding)

## Introduction

High availability is crucial in High-Performance Computing (HPC) environments, where unexpected failures can constrain system performance. Solid-state drives (SSDs) are essential for data-intensive applications due to their high speed and are often used as dedicated or node-local burst buffers (BBs). However, the limited write endurance of SSDs, which varies with utilization, requires thorough investigation to prevent unforeseen failures that could jeopardize scientific computations.

In this study, we propose a hierarchical modeling approach to evaluate the reliability and performability of BBs. We developed a model powered by machine-learning algorithms to dynamically predict storage failures based on wear caused by different applications. Two models, based on generalized stochastic Petri nets (GSPNs) and reliability block diagrams (RBDs), were created to represent and evaluate BBs. Additionally, we conducted an exploratory study using a representative dataset to analyze the impact of workload on SSD failures. Benchmarks from the MOGON II supercomputer demonstrate the feasibility of our approach. This project is developed within the [IO-SEA project](https://iosea-project.eu) for Exascale Storage I/O and Data Management.

Within this repository, you'll find Jupyter notebooks dedicated to analyzing SSD failures and performance using advanced machine learning techniques. It also includes a comprehensive exploration into the evolution of intrinsic wear-related characteristics and failures in HDDs, conducted through detailed exploratory data analysis based on their utilization (number of written blocks). Moreover, the repository offers a practical guide on harnessing MongoDB to efficiently store and analyze large, cleaned datasets, thereby enhancing data access performance.

## Objectives

- Performed an explanatory analysis of an industry dataset from Alibaba to investigate the impact of workloads on SSD failures. We aim to understand the distinct effects that workloads may have on these storage technologies.
- Formulated two analytical models based on the mathematical formalisms RBD (reliability block diagrams) and GSPN (generalized stochastic Petri nets) to evaluate the reliability and performability of burst buffer systems. These models enabled us to represent the respective nodes composing such systems and also estimate system mean time to failure, reliability and throughput.
- Conceived a model called Dynamic Mean Time to Failure (DMTTF) for dynamically estimating SSD failures by using machine-learning algorithms. The DMTTF model is based on an evaluation of wear progress, which considers the number of written blocks on a storage device at a given time, along with aging-related internal sensor values.
- Carried out a reliability and performability study adopting our proposed DMTTF, RBD, and GSPN models. Experiments were performed based on HPC applications (LQCD and ECMWF) to demonstrate the feasibility of the proposed approach utilizing the MOGON II supercomputer and GekkoFS benchmarks.
- Conducted exploratory data analysis using SMART statistics and failure logs from a Backblaze dataset to evaluate the evolution of HDD failures and failure-related attributes in relation to device usage, particularly the number of written blocks.

## Methodology

#### Proposed Method

This section summarizes our methodology for modeling and evaluating burst buffer (BB) systems in high-performance computing environments, considering the unique characteristics of different applications.

#### Modeling Burst Buffer Systems

We aim to estimate BB systems' dependability and performability. The methodology involves several key steps: 

1. **Input**: The system architecture description, including the BB layer and dependability relationships, allows the creation of abstract models. This step involves collecting SSD SMART values related to wear and the number of blocks written by applications to predict SSD failures accurately.

2. **Machine Learning Model**: We utilize a machine learning-based dynamic mean time-to-failure (DMTTF) model to predict SSD failures. This model analyzes the SMART attributes and workload characteristics to dynamically estimate failure rates, which are crucial for accurate reliability and performability assessments.

3. **Modeling**: We design and refine GSPN and RBD models to represent the system. The machine learning model's predictions are integrated into these models to reflect real-world performance and failure metrics. This step includes creating abstract models and tuning them to align with observed data.

4. **Evaluation**: The models are evaluated and refined based on experimental results. We compute the reliability and performability models to predict BB failures and analyze system performance under failure conditions.

#### Hierarchical Modeling

Our hierarchical modeling approach investigates BB failures' impact on performance. The system architecture is presented generically, applicable to both single nodes and pools of BB nodes.

1. **Input**: SMART values and workload characteristics essential for DMTTF learning and failure prediction.

2. **MTTF Prediction**: The machine learning model calculates the annual failure rate (AFR) and MTTF, which are used as parameters in the reliability model.

3. **Dependability Evaluation**: Calculate the MTTF and mean time to repair (MTTR) for the entire system, considering BB dependability without redundancy mechanisms.

4. **Performability Evaluation**: Assign MTTF and MTTR to transitions in the GSPN model to estimate system performance under failure conditions.

By following this methodology, valuable insights for data-driven decision-making can be obtained, improving BB system performance and preventing failures through proactive measures.

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

## Real traces

Lattice quantum-chromodynamics (LQCD) and European Centre for Medium-Range Weather Forecasts (ECMWF). The former, LQCD, is a prescription for understanding how quarks and gluons interact to give rise to the properties of composite particles such as protons, neutrons, and mesons. The latter comprises a time-critical global numerical weather forecast. We only considered write requests, as these are the focus of this study.

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
     - **Approach**: Create collections using `db.createCollection`. View existing collections with `show collections`.
   
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

This notebook outlines the setup of MongoDB for handling SSD and HDD data, including data population, basic operations, querying, and performance optimization strategies. These steps are crucial for effectively managing and analyzing large datasets in MongoDB.

### Data Processing and Exporting for HDD
This notebook focuses on methods for processing and exporting HDD data.

**Content**:
1. **Extracting HDD Metrics**:
   - **Key Metrics**: Extraction of critical metrics such as failure rates, reallocated sectors, uncorrectable errors, number of blocks written, command timeouts, uncorrectable sector count, current pending sector, and timestamps from specific HDD models stored in a MongoDB database.

2. **Data Processing Techniques**:
   - **Data Cleaning**: Removing noise and irrelevant data points for clarity.
   - **Normalization**: Scaling data to ensure consistency across analyses.
   - **Feature Selection**: Identifying and selecting important metrics for further analysis.

3. **Data Export Methods**:
   - **Efficient Data Export**: Techniques for seamless integration with analytical tools.
   - **Common Formats**: Exporting data into CSV files for subsequent exploratory analysis.

4. **Tools and Frameworks**:
   - **Python Libraries**: Utilizing pandas for robust data manipulation.
   - **MongoDB**: Efficiently storing and querying large datasets.

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
This notebook provides an in-depth exploration of SSD failures across various applications, analyzing failure rates such as AFR (Annualized Failure Rate) and MTTF (Mean Time To Failure) in relation to SSD characteristics.

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

This notebook provides a comprehensive exploration of SSD failures and performability metrics across various applications, leveraging machine learning models to enhance the analysis. The objective is to understand the factors affecting SSD reliability and improve their usage in burst buffer tiers within the high-performance computing field.

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
     - Merge datasets to associate SSD models with failure logs.
     - Loop through each application and SSD model to calculate AFR and MTTF.
     - Compile statistics to track the evolution of failures in SSDs and wear-related smart attributes (such as reallocated sectors count and wear leveling) in relation to the number of written blocks per application.

5. **Machine Learning Model Development**
   - **Objective**: Develop and validate machine learning models to predict SSD failures.
   - **Approach**:
     - Split the data into training and testing sets.
     - Train various machine learning models, including Random Forest, LSTM, and XGBoost.
     - Perform hyperparameter tuning to optimize model performance.     

6. **Model Validation and Performance Evaluation**
   - **Objective**: Validate the performance of the selected machine learning model.
   - **Approach**:
     - Perform cross-validation to ensure model robustness.
     - Evaluate models using statistical measures: mean absolute error, mean squared error, and r-squared
     - Interpret model predictions and identify key features influencing SSD failures.
     - Select the best-performing model for further analysis.

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

This notebook offers utility functions to assist in calculating system reliability metrics using Reliability Block Diagram (RBD) analysis and in approximating delay distributions with phase-type distributions.

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

## Publication
A scientific paper based on the results from the notebooks was authored by Eric Borba, Reza Salkhordeh, Salim Mimouni, Eduardo Tavares, Paulo Maciel, Hossein Asadi, and André Brinkmann. It was published in the Proceedings of the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS), held in Potsdam, Germany, from May 14th to 16th, 2024.

- **Title**: A Hierarchical Modeling Approach for Assessing the Reliability and Performability of Burst Buffers
- **Abstract**: High availability is a crucial aspect of High-Performance Computing. Solid-state drives (SSD) offer peak bandwidth as node-local burst buffers. The limited write endurance of SSDs requires thorough investigation to ensure computational reliability. We propose a hierarchical model to evaluate the reliability and performability of burst buffers. We developed a machine-learning model to dynamically predict storage failures according to the wear caused by different applications. We also conducted an exploratory study to analyze the workload effects on SSD failures, and a representative dataset was adopted.
- **Presentation**:
The paper was presented by Eric Borba at the 37th GI/IT International Conference on Architecture of Computing Systems (ARCS).

## Fundings

This work has been funded by the European Union’s Horizon 2020 JTI-EuroHPC research and innovation programme and the BMBF/DLR under the "IO-SEA" project with grant agreement number: 955811. This work was partially supported by Conselho Nacional de Desenvolvimento Científico e Tecnológico – CNPq under grant 202998/2019-3.