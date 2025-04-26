Project description:

Data used for Analysis: Bank data

Outcome: To predict if the client will subscribe for a bank term deposit (variable y)

Methodology:
1. Convert categorical into dummies
2. Feature selection
3. Predict the target variable (classes) using Logistic Regression, Random Forest, Gradient Boosting/XGBoosting, Neural network models
4. Finalise the model with high predictive power
5. Save the predictions in csv format, save plots if any 
6. Predictions.ipynb is our file but the script has to be split into 3 scripts:
   Data ingestion and pre-processing: pipelines.py, 
   Train the data and building model: modelling.py
   Call the main script that executes all other scripts: main.py
7. Create a test.py script to test the main.py
8. Push it to Github repo, create yaml file and CI/CD
9. Ready for deployment as an API and on AzureML (track hyperparameters, metrics), deploy the model as webservice and serverless 
