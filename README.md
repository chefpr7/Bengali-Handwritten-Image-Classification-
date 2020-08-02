# Bengali Handwritten Image Classification
[Kaggle Competition] **Given images of handwritten bengali images, the challenge was to separate the graphemes, the vowel diacritics and the consonant diacritics from the images** 

# Demo 
<div align = "center" >
<img src='https://github.com/chefpr7/Bengali-Handwritten-Grapheme-Classification-/blob/master/ezgif.com-crop.gif'/>
<br></br>
Link to full video : 
</div>

# Table of Contents:-
* [Problem Statement](#problem-statement)
* [About the Dataset](#about-the-dataset)
* [How to get the webapp running for you?](#how-to-get-the-webapp-running-for-you)
* [Solution Approach](#solution-approach)
* [Important Links and Requirements](#important-links-and-requirements)

## Problem Statement 
*Bengali is the 5th most spoken language in the world with hundreds of million of speakers. It’s the official language of Bangladesh and the second most spoken language in India. Considering its reach, there’s significant business and educational interest in developing AI that can optically recognize images of the language handwritten. This challenge hopes to improve on approaches to Bengali recognition.<br></br>
Optical character recognition is particularly challenging for Bengali. While Bengali has 49 letters (to be more specific 11 vowels and 38 consonants) in its alphabet, there are also 18 potential diacritics, or accents. This means that there are many more graphemes, or the smallest units in a written language. The added complexity results in ~13,000 different grapheme variations (compared to English’s 250 graphemic units).*
<div align="center">
<img src='https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F1095143%2Fa9a48686e3f385d9456b59bf2035594c%2Fdesc.png?generation=1576531903599785&alt=media' width=400 height=400 />
</div>

**For this competition, you’re given the image of a handwritten Bengali grapheme and are challenged to separately classify three constituent elements in the image: grapheme root, vowel diacritics, and consonant diacritics.**

## About the Dataset
 The training contained around 0.2 million images origininally supplied as binary vectors in parquet format which were converted to .jpg format for ease. The dimension for every image was originally 137 X 236 . The test set was of approximately same size as that of the training set but it contained several graphemes which had not been supplied in the training set, however the graphemes contained the combination of same roots, vowel diacritics and consonant diacritics. Further information on the dataset can be found below in the list of impotant links.
 
## How to get the webapp running for you
 * Fork the repo 
 * Open [web-app-runner](https://github.com/chefpr7/Bengali-Handwritten-Grapheme-Classification-/blob/master/bengali_web_app_runner.ipynb)
 * Add the models in your Google drive from the list of important links (Use proper directories, in case of any changes edit line 69, 88, 120 & 122 of app.py with the new paths)
 * Run the web-app runner and get going 

## Solution Approach 
 * Trained several models of **Resnet50, Resnet101, efficientnet b0, b1, b2, b3, SeREsNEXT** on Pytorch
 * **Ensembled** over the best performing models 
 * Tried **Cutmix, Cutout, Fmix** - new augmentaion techniques to achieve better generalization and results 
 * Trained **3 separate models for grapheme root, vowel diacritic and consonant diacritic** respectively 
 * Trained **1 common model for the entire task treating it as a multi-class classification task** 
 * Performed **Test Time Augmentations** to get better results
 * **Metric used** : *F1 score* 
 * **Loss Function** : *BCE with Logits*, *Categorical Crossentropy* 
 * **Optimizers** : *Adam*, *Ranger*, *SGD*
 * **Best Recall Score on Public leaderboard** : *0.9690*
 * **Best Recall Score on Private leaderboard** : *0.9474*
 * **Position** : *229/2059* participating teams 
 
## Important Links and Requirements
 * Resnet50 trained model with 186 final fc layers - https://drive.google.com/file/d/1hLOLf92WE5fk8YzXiRYIqvSa7DhwlJOj/view?usp=sharing
 * Link to dataset of images in .jpg format : https://drive.google.com/file/d/1G0wxfphIbQtmtzpqbEPcWdRI2kuNuDPh/view?usp=sharing
 * Link to competition : https://www.kaggle.com/c/bengaliai-cv19
 * Link to data : https://www.kaggle.com/c/bengaliai-cv19/data
 
