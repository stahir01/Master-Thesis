# Master Thesis Project Repository

This repository serves as a comprehensive archive, encompassing all aspects of my research. Here, you will find detailed documentation of the research papers reviewed, datasets utilized, and insights into the machine learning models developed and evaluated throughout my study.


## Literature Review

In the `literature_review` directory, you'll find summaries and analyses of all research papers read during the study. Each paper's summary includes its main findings, methodologies, and how it contributed to this research project.

## Datasets

The `datasets` directory contains all the data collected or used in this project. Here, you will find:

- Raw data files
- Descriptions of each dataset
- Scripts used for data preprocessing
- Links to external data sources (if applicable)

## Machine Learning Models

The `models` directory is dedicated to the machine learning models developed and tested in this project. It includes:

- Code for each model
- Detailed explanations of the algorithms used
- Configuration files for model parameters
- Evaluation metrics and results

## Results

This section presents the outcomes of the research, including:

- Comparative analysis of different models
- Graphs and charts illustrating the results
- Interpretations and conclusions drawn from the data analysis


## Next Steps
As the research progresses, the following steps will be executed to further refine the model and extend the scope of the project:

#### Task 1: Refine Input Parameters:

- Focus on key parameters that have the most significant impact on predicting aircraft range. This involves removing less important features such as **Fuselage** **Length**, **Fuselage Width**, **Wing Taper Ratio**, **Wing Sweep**, and **Aspect Ratio**.
Emphasize features like **Max Usable Fuel**, **Payload**, **V_MO**, and **ln(Initial Weight/Final Weight)**, **Max Takeoff Weight**, **Operating Empty Weight**, **Max Zero Fuel Weight**, **Wing Area**, for better model performance.


#### Task 2: Perform Simulations and Sensitivity Analysis
- Conduct simulations to evaluate the sensitivity of the model. In these simulations, keep most parameters constant and vary a few key features (e.g., payload or speed) to assess their impact on predicted range. This will validate the robustness of the model.


#### Task 3:  Implement Reverse Prediction:

- Develop a reverse prediction approach where the model is provided with a target range and some known parameters (like payload and V_MO) to predict missing values such as fuel capacity or total aircraft weight.
This will add a practical component to the research, allowing real-world applications such as optimizing payload or fuel based on a desired range.


#### Task 4: Review Recorded Video


#### Task 5: Work on Simualtion data: