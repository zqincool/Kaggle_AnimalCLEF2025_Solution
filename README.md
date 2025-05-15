# AnimalCLEF2025 - Animal Re-identification Challenge

This project aims to solve the animal re-identification task in the AnimalCLEF2025 challenge. The task requires identifying individuals from the following three species:
1. Loggerhead sea turtles
2. Salamanders
3. Eurasian lynxes

## Dataset Structure
The dataset contains three main folders:
- SeaTurtleID2022/
- SalamanderID2025/
- LynxID2025/

Each species folder contains:
- database/: training set
- query/: test set

## Evaluation Metrics
- BAKS (Balanced Accuracy on Known Samples): Balanced accuracy for known individuals
- BAUS (Balanced Accuracy on Unknown Samples): Balanced accuracy for unknown individuals
- Final Score: Geometric mean of BAKS and BAUS

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Ensure the dataset is correctly placed in the `images` folder
2. Copy metadata.csv to specified model folder
3. Run the model:
- Example:
```bash
cd All_Species
cd Swin_Transformer
python swinTransformerClassifier.py
``` 