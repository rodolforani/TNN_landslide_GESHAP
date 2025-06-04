# The repository encmpass the workflow of our modeling, diveded in separates codes

## 1. Training
- Folder "params" >> inside params.json modify all the pamarameters for the TNN and input files.
- Folder "params" >> inside paramsTerrain.json modify all the pamarameters for the DNN, so if only static features.
- ATTENTION: set inference false for the training phase, and true after for the predictions. 
- Run TransformerModels.ipynb for train the model
## 2. Testing
- Set Inference in true in params and then run EvalBenchmark.ipynb
## 3. Explanation
- Inside ExplainModels.ipynb youc can find different approaches to retrive the explainability scores, we selected the clearest one (everthing is commented inside the code)
## 4. Plot the results 
- Use the remaining codes fro some plots
