# Olfaction-Intelligence

### OlfecNet: Intelligence

## 📌 Project Overview  
**OlfecNet - Intelligence** is a neural network-based solution designed for odor perception prediction. Leveraging machine learning, it analyzes molecular structures and GC-MS data to predict odor characteristics with improved accuracy.

This project focuses on building an end-to-end pipeline for data preprocessing, feature extraction, model training, and evaluation for odor recognition tasks.



## 📂 Project Structure  
```
OlfecNet-Intelligence/
├── OlfecNet-Intelligence.ipynb    # Main Jupyter notebook for the workflow
├── data/                         # Raw and processed datasets (molecular, GC-MS, odor labels)
├── models/                       # Saved models and checkpoints
├── results/                      # Generated results and evaluation metrics
├── README.md                     # Project documentation
```

---

## 🚀 Features  
- GC-MS data preprocessing  
- Molecular feature extraction  
- Neural network model for odor classification  
- Performance evaluation (accuracy, loss curves, confusion matrix)  
- Reproducible notebook with step-by-step explanation  

---

## 🛠️ Technologies Used  
- Python  
- PyTorch 
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib / Seaborn for visualization  

---

## 🧪 How to Run  

1. Clone the repository:  
```bash
git clone https://github.com/yashnayi234/Olfaction-Intelligence.git
cd OlfacNet-Intelligence
```

```bash
pip install -r requirements.txt
```


## Results and Evaluation

Accuracy: 99.05%        
Loss: 0.0371

## Future Work
 - Expand the dataset with more molecular samples
 - Implement hyperparameter tuning (Grid Search / Optuna)
 - Explore transformer-based architectures for sequence modeling
