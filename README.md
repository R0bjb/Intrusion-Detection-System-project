# Hierarchical Machine Learning for Network Intrusion Detection

## Table of Contents

- [Overview](#overview)
- [Built With](#built-with)
- [Features](#features)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)

## Overview
A Hierarchical machine learning pipeline for multiclass DDoS attack classification, designed to improve interpretability, accuracy and training efficiency over flat classification approaches.
This project resulted in improved interpretability and efficiency, resulting in training times around 1 minute faster (~50%), despite resulting in similar accuracy results compared to flat approaches (roughly 63%)
The approach split the problem into layers, each falling into the next:
    Binary - The first layer simply classified whether the data was an attack or not
    Group - The following layer split attacks into similar attack groups.
    individual - Following the group stage, the attack is classified only as an attack that belongs to the group.
The project provided a great opportunity to develop skills, giving technical lessons in machine learning, cyber security, data analysis and software development, whilst also developing skill including problem solving, written communication, time management and perserverance.
    
Within this file resides the programming for my university dissertation project. 
the following program explores a unique approach towards intrusion detection through the use of machine learning, specifically a hierarchical approach that simplifies the classification by splitting into group classification, starting at a binary level, leading to broad group classification, to individual classification within each group. 
the resulting program worked successfully, though not affecting accuracy scores, providing further interpretability which was not possible previously, whilst improving efficiency, reducing training times significantly.


<!-- TODO: Add a screenshot of the live project.
    1. Link to a 'live demo.'
    2. Describe your overall experience in a couple of sentences.
    3. List a few specific technical things that you learned or improved on.
    4. Share any other tips or guidance for others attempting this or something similar.
 -->

### Built With
Python
numpy    https://numpy.org/
sklearn    https://scikit-learn.org/stable/
pandas    https://pandas.pydata.org/docs/user_guide/index.html
Matplotlib    https://matplotlib.org/
Seaborn    https://seaborn.pydata.org/

to run the program you will be required to:
1) download the CICDDoS2019 dataset
2) set file path in main.py to align with the location of CICDDoS2019 dataset
3) run python main.py

<!-- TODO: List any MAJOR libraries/frameworks (e.g. React, Tailwind) with links to their homepages. -->

## Features
The program explores methods to improve detection and multiclass classification on various types of DDoS attacks on a network. 
The resulting model provides greater interpretability and improved efficiency compared to previous models, however regarding accuracy the system remains on par with standard multiclass classificaiton setups.


<!-- TODO: List what specific 'user problems' that this application solves. -->

## Contact
www.linkedin.com/in/robert-blakeley-611176257



<!-- TODO: Include icons and links to your RELEVANT, PROFESSIONAL 'DEV-ORIENTED' social media. LinkedIn and dev.to are minimum. -->

## Acknowledgements

<!-- TODO: List any blog posts, tutorials or plugins that you may have used to complete the project. Only list those that had a significant impact. Obviously, we all 'Google' stuff while working on our things, but maybe something in particular stood

