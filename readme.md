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
AutoGluon is an auto machine learning tool. It trains base models of KNeighborsUnif, KNeighborsDist, NeuralNetFastAI, LightGBMX, LightGBM, RandomForestGini, RandomForestEntr, CatBoost, ExtraTreesGini, ExtraTreesEntr, XGBoost, NeuralNetTorch, LightGBMLarge, fine tune hyperparameters automatically, and select the model with the best performace. It also uses a Greedy Weighted Ensemble algorithm to combine the outputs of multiple base models to produce a final prediction.

Besides AutoGluon, we also train SVM, CNN, MLP, and Transformer. We use early stop to get the appropriate epoch.

For each model, we train using multidal data (text + image), image data only, and text data only, respectively.

## Evaluation Metrics
We use F1, recall, precision, and accuracy.

## Result
The detailed reports be refered at ```./circl_phishing_dataset/classification_report_{model_name}_{data}.csv```

.|Accuracy|Precision|Recall|F1
-|-|-|-|-
AutoGluon (Multi)|0.6153846153846154|0.6627658056229485|0.6153846153846154|0.6202202559345417
AutoGluon (Image)|0.5934065934065934|0.5307552297996636|0.5934065934065934|0.5471171620141871
AutoGluon (Text)|0.5384615384615384|0.5707066742781028|0.5384615384615384|0.5314354432001491
CNN (Multi)|0.7717391304347826|0.7601319875776397|0.7717391304347826|0.7488412979461572
CNN (Image)|0.7282608695652174|0.7178947863730472|0.7282608695652174|0.684295318918081
CNN (Text)|0.6847826086956522|0.6772891963109354|0.6847826086956522|0.6549385772615682
MLP (Multi)|0.7717391304347826|0.7656099033816425|0.7717391304347826|0.7480891994478951
MLP (Image)|0.7391304347826086|0.7154589371980676|0.7391304347826086|0.7162460045040137
MLP (Text)|0.695652173913043|0.6794254658385093|0.6956521739130435|0.6752547395769902
SVM (Multi)|0.7391304347826086|0.7262516469038208|0.7391304347826086|0.7020174156619169
SVM (Image)|0.7391304347826086|0.710144927536232|0.7391304347826086|0.7032042061675928
SVM (Text)|0.6739130434782609|0.6265010351966874|0.6739130434782609|0.6312493665754535
<b>Transformer (Multi)|0.7934782608695652|0.7778252242926157|0.7934782608695652|0.7734424427519057</b>
Transformer (Image)|0.7608695652173914|0.7613185425685426|0.7608695652173914|0.747616061474757
Transformer (Text)|0.717391304347826|0.6962603519668739|0.717391304347826|0.6938017598343685




