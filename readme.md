# Coursework for CSC 8851 DEEP LEARNING

## Introduction
This project focuses on developing a *reference-based* phishing webpage detection system utilizing webpage screenshots as input.

Phishing webpages are identified as those that mimic the appearance of a legitimate brand while having a domain name that does not correspond to that brand's official domain. Consequently, this project involves a two-step approach:

1. Brand Appearance Classification:

In the first step, the system classifies whether a given webpage contains visual elements or features that are associated with known brands.

2. Domain Verification for Phishing Detection:

In the second step, for webpages identified as containing known brand appearances, the system verifies whether the webpage's domain name matches the legitimate domain associated with that brand.
If a mismatch is detected, the webpage is classified as a phishing webpage.

This framework combines visual analysis of webpage content with domain verification to provide an effective solution for detecting phishing webpages based on their design and URL inconsistencies. 

**As it is a Deep Learning course, we only construct the first step here.**

## Data

This data set is from [circl-phishing-dataset-01](https://www.circl.lu/opendata/datasets/circl-phishing-dataset-01/), see Introduction [here](https://www.circl.lu/opendata/circl-phishing-dataset-01/). It contains 457 phishing webpage screenshots with inddicating their mimiced brands.

Images are at ```./circl_phishing_dataset/Clean_phishing/```
Labels (brands) are at ```./circl_phishing_dataset/*.json ```

## Preprocessing
1. We use OCR to extract textual content from the screenshots.
2. We clean the textual content that we extracted by removing newlines, tabs, punctuations, and excessive spaces

3. We convert the image and text to embeddings via [CLIP (Contrastive Language-Image Pre-Trainin)](https://github.com/openai/CLIP)

## Training 
**AutoGluon** is an auto machine learning tool. It trains base models of KNeighborsUnif, KNeighborsDist, NeuralNetFastAI, LightGBMX, LightGBM, RandomForestGini, RandomForestEntr, CatBoost, ExtraTreesGini, ExtraTreesEntr, XGBoost, NeuralNetTorch, LightGBMLarge, fine tune hyperparameters automatically, and select the model with the best performace. It also uses a Greedy Weighted Ensemble algorithm to combine the outputs of multiple base models to produce a final prediction.
```
**KNeighborsUnif**: K-Nearest Neighbors (Uniform Weights)
**KNeighborsDist**: K-Nearest Neighbors (Distance Weights)
**NeuralNetFastAI**: Neural Network using the FastAI framework
**LightGBMX**: Light Gradient Boosting Machine (Custom Hyperparameters X)
**LightGBM**: Light Gradient Boosting Machine (Default Parameters)
**RandomForestGini**: Random Forest Classifier (Gini Impurity Criterion)
**RandomForestEntr**: Random Forest Classifier (Entropy Criterion)
**CatBoost**: Categorical Boosting (CatBoost) Model
**ExtraTreesGini**: Extremely Randomized Trees (Gini Impurity Criterion)
**ExtraTreesEntr**: Extremely Randomized Trees (Entropy Criterion)
**XGBoost**: Extreme Gradient Boosting (XGBoost) Model
**NeuralNetTorch**: Neural Network using PyTorch
**LightGBMLarge**: Light Gradient Boosting Machine (Larger Model)
```

Besides AutoGluon, we also train **SVM** (Support Vector Machine), **CNN** (Convolutional Neural Network), **MLP** (Multi-Layer Perceptron), and **Transformer**. We use early stop to get the appropriate epoch.

For each model, we train using multidal data (text + image), image data only, and text data only, respectively.

## Evaluation Metrics
We use F1, recall, precision, and accuracy.

## Result
The detailed reports be refered at ```./circl_phishing_dataset/classification_report_{model_name}_{data}.csv```

.|Accuracy|Precision (Weighted)|Recall(Weighted)|F1 (Weighted) |Precision (Macro)|Recall(Macro)|F1 (Macro)
-|-|-|-|-|-|-|-
AutoGluon (Multi)|0.6153846153846154|0.6627658056229485|0.6153846153846154|0.6202202559345417 |0.4296148296148296|0.39146723646723647|0.39257298257298257
AutoGluon (Image)|0.5934065934065934|0.5307552297996636|0.5934065934065934|0.5471171620141871|0.3691626055104315|0.377094017094017|0.3659551764787463
AutoGluon (Text)|0.5384615384615384|0.5707066742781028|0.5384615384615384|0.5314354432001491|0.36640769944341367|0.32673992673992674|0.3340367678602973
CNN (Multi)|0.7717391304347826|0.7601319875776397|0.7717391304347826|0.7488412979461572| 0.712313988095238|0.7645499465811966|0.7178789143862674
CNN (Image)|0.7282608695652174|0.7178947863730472|0.7282608695652174|0.684295318918081|0.6671289494818907|0.7277526395173454|0.6751544591509989
CNN (Text)|0.6847826086956522|0.6772891963109354|0.6847826086956522|0.6549385772615682|0.6466866466866467|0.678571428571428|0.6429159518058236
MLP (Multi)|0.7717391304347826|0.7656099033816425|0.7717391304347826|0.7480891994478951|0.6719618055555556|0.7298277243589744|0.6809590583028082
MLP (Image)|0.7391304347826086|0.7154589371980676|0.7391304347826086|0.7162460045040137|0.6055555555555555|0.6656241906241906|0.6246519075466443
MLP (Text)|0.695652173913043|0.6794254658385093|0.6956521739130435|0.6752547395769902|0.5729910714285715|0.6402711004273505|0.5881856909430438
SVM (Multi)|0.7391304347826086|0.7262516469038208|0.7391304347826086|0.7020174156619169|0.6843893480257117|0.7268842268842269|0.6837127578304047
SVM (Image)|0.7391304347826086|0.710144927536232|0.7391304347826086|0.7032042061675928|0.6957070707070707|0.7496114996114996|0.7064332548543074
SVM (Text)|0.6739130434782609|0.6265010351966874|0.6739130434782609|0.6312493665754535|0.6224664224664225|0.6437451437451438|0.6191036236490782
<b>Transformer (Multi)</b>|<b>0.7934782608695652</b>|<b>0.7778252242926157</b>|<b>0.7934782608695652</b>|<b>0.7734424427519057</b>|<b>0.7478546626984126</b> |<b>0.7884214743589744</b> |<b>0.7547530390361272</b>
Transformer (Image)|0.7608695652173914|0.7613185425685426|0.7608695652173914|0.747616061474757|0.710571112914863|0.7305221688034188|0.7061441163003663
Transformer (Text)|0.717391304347826|0.6962603519668739|0.717391304347826|0.6938017598343685|0.6384300595238095|0.6914863782051281|0.644233630952381




