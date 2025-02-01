# 🌱 SepalSense

SepalSense is a machine learning project that classifies Iris flower species based on sepal and petal dimensions. This repository implements a **Random Forest Classifier** to predict the species of an iris flower using the famous **Iris dataset**.

---

## 📌 Features
- Trains a **Random Forest Classifier** on the Iris dataset
- **Feature selection**: Sepal & Petal length/width
- **Model serialization** using Pickle for easy deployment
- Simple and clean **Python implementation**

---

## 📂 Project Structure
```
SepalSense/
│-- iris.csv          # Dataset file
│-- model.py         # Python script for training the model
│-- model.pkl        # Serialized trained model
│-- README.md        # Project documentation
```

---

## 🚀 Installation & Setup

1. **Clone the repository**:
```sh
 git clone https://github.com/ad1lhasan/SepalSense.git
 cd SepalSense
```

2. **Install dependencies**:
```sh
 pip install -r requirements.txt
```
(Ensure `pandas` and `scikit-learn` are installed.)

3. **Run the model training script**:
```sh
 python model.py
```
This will train the model and save it as `model.pkl`.

---

## 📊 Dataset
The project uses the classic **Iris dataset**, which contains **150 samples** of iris flowers with 4 features:
- **Sepal Length**
- **Sepal Width**
- **Petal Length**
- **Petal Width**

The dataset consists of three classes:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

---

## 🛠 Technologies Used
- Python 🐍
- Pandas 📊
- Scikit-Learn 🤖
- Random Forest Classifier 🌲

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 🤝 Contributing
Contributions are welcome! Feel free to open an issue or submit a pull request.

---

## 📬 Contact
For questions or suggestions, reach out via [muhammedadilhasan@gmail.com or ad1lhasan].

Happy Coding! 🚀

