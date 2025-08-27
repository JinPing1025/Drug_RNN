# 🏕 Drug_RNN
A molecular generation model based on deep learning algorithm. The generation task is implemented in the TensorFlow framework, allowing the user to run the model to generate a focused library of drug-like molecules.

# ⚙ Requirement
```
Refer to requirement.txt
```

# 🔧 Installation
* Install [python 3.7](https://www.python.org/downloads/) in Linux or Windows.
* If you want to run on a GPU, you will need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), please refer to their websites for corresponding versions.
* Add the installation path and run the following command to install all the environment libraries in one step.
```
pip install -r requirement.txt
```

# ✈ Running this Model
You need to open main.py, run load_weights to read the pre-trained weights and get the generated molecules.
Or provide training set molecules into coding for model training.

# 💥 Dataset
Folder datasets contains pre-training or transfer learning molecules, folder generate contains the molecules generated after transfer learning, folder gen_data is based on the generation model or machine learning model screening molecules.

# 📖 Article
For more details about the methodology and experimental results, please refer to my article:  
[👉 Click here to view the paper：10.1007/s00894-023-05772-5]
