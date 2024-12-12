# **Titanic Neural Network Tutorial**

This project implements a **Feedforward Neural Network** using TensorFlow/Keras to predict survival on the Titanic dataset. The tutorial focuses on binary classification and includes detailed steps for data preprocessing, model building, training, and evaluation.

---

## **Project Overview**

This tutorial demonstrates a binary classification task to predict whether a passenger survived the Titanic disaster based on features such as **age**, **gender**, **fare**, and **class**. The model achieves an accuracy of **81%** on the test data.

---

## **Project Structure**

The repository contains the following files:

| **File**                     | **Description**                                        |
|------------------------------|--------------------------------------------------------|
| `titanic_neural_network.ipynb` | Jupyter Notebook with code and explanations.          |
| `Titanic-Dataset.csv`        | Input dataset for training and testing.               |
| `titanic_tutorial.pdf`       | Detailed tutorial explaining theory and implementation.|
| `README.md`                  | This file describing the project.                     |

---

## **Dataset Description**

The Titanic dataset includes passenger information to predict survival outcomes. The key features are:

| **Feature**     | **Description**                              |
|------------------|----------------------------------------------|
| `Pclass`        | Passenger class (1 = First, 2 = Second, 3 = Third) |
| `Sex`           | Gender of the passenger                     |
| `Age`           | Age of the passenger                        |
| `SibSp`         | Number of siblings/spouses aboard           |
| `Parch`         | Number of parents/children aboard           |
| `Fare`          | Ticket fare                                 |
| `Embarked`      | Port of embarkation                         |
| `Survived`      | Target variable (1 = Survived, 0 = Not Survived) |

---

## **Steps to Run the Project**

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/titanic_neural_network_tutorial.git
cd titanic_neural_network_tutorial

---

## **Steps to Run the Project**

Follow these steps to set up and run the project:

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/titanic_neural_network_tutorial.git
cd titanic_neural_network_tutorial
```

### 2. Install the Required Libraries

Install the necessary Python libraries using pip:

```bash
pip install tensorflow pandas scikit-learn matplotlib numpy
```

### 3. Launch Jupyter Notebook

Open the Jupyter Notebook environment and launch the project:

```bash
jupyter notebook titanic_neural_network.ipynb
```

### 4. Run the Notebook

Run all cells in the notebook to perform the following tasks:

- **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
- **Model Building**: Define and compile the neural network model.
- **Training**: Train the model on the training dataset.
- **Evaluation**: Evaluate the model on the test dataset.

---

## **Neural Network Model**

The neural network consists of the following architecture:

- **Input Layer**: 64 neurons with ReLU activation  
- **Hidden Layer**: 32 neurons with ReLU activation  
- **Dropout Layer**: 30% dropout for regularization  
- **Output Layer**: 1 neuron with Sigmoid activation  

### **Model Code**

```python
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
```

---

## **Results**

The model was evaluated on the test dataset. Below are the key performance metrics:

| **Metric**       | **Class 0 (Not Survived)** | **Class 1 (Survived)** |
|-------------------|----------------------------|------------------------|
| **Precision**     | 0.82                       | 0.79                   |
| **Recall**        | 0.87                       | 0.73                   |
| **F1-Score**      | 0.84                       | 0.76                   |

- **Overall Accuracy**: **81%**

### **Training and Validation Loss Plot**

Add the following image to visualize the loss:

```markdown
![Training and Validation Loss](insert_training_loss_plot_here.png)
```

---

## **Key Features**

1. **Data Preprocessing**:  
   - Handling missing values for `Age` and `Embarked`.  
   - One-hot encoding for categorical variables like `Sex` and `Embarked`.  
   - Feature scaling for `Age` and `Fare` using StandardScaler.  

2. **Model Architecture**:  
   - Input and hidden layers with ReLU activation.  
   - Dropout regularization to prevent overfitting.  

3. **Evaluation Metrics**:  
   - Accuracy, Precision, Recall, and F1-Score.  

4. **Visualization**:  
   - Plots for training and validation loss.

---

## **Future Improvements**

- **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV to optimize model parameters.  
- **Batch Normalization**: Add batch normalization for faster convergence.  
- **Ensemble Methods**: Test ensemble techniques like Random Forests or XGBoost for comparison.

---

## **How to Use This Project**

- Run the **Jupyter Notebook** for step-by-step execution.  
- Use the saved model file `titanic_neural_network.h5` to make predictions on new data.  
- Refer to `titanic_tutorial.pdf` for a detailed explanation of the workflow and theory.

---

## **License**

This project is licensed under the MIT License. You are free to use, modify, and distribute the code for educational or commercial purposes.

---

## **References**

1. Titanic Dataset: [Kaggle - Titanic Competition](https://www.kaggle.com/c/titanic/data)  
2. TensorFlow Documentation: [TensorFlow.org](https://www.tensorflow.org/)  
3. Scikit-learn Documentation: [Scikit-learn.org](https://scikit-learn.org/)

---

## **Contact**

For any questions or suggestions, feel free to contact me:

- **Name**: Gopi Erla  
- **Email**: erlagopi05@gmail.com  
- **GitHub**: [https://github.com/GopiErla]
```
