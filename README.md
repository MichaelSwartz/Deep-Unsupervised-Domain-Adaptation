# DDC-specifc Fork of Deep-Unsupervised-Domain-Adaptation

This is a fork of [Deep-Unsupervised-Domain-Adaptation](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation), which provided a more-up-to-date version of the code originally developed in (DDC-transfer-learning
)[https://github.com/syorami/DDC-transfer-learning].

- Paper for that repo: [Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition](https://arxiv.org/abs/2109.13420)
- Paper for the original repo: [DDC](https://arxiv.org/abs/1412.3474)

## Setup 
---
- This project and requirements were built on a Linux machine, and is not guaranteed to work on a different operating system. It is optimized for CUDA-enabled GPUs, but can be run on typical CPUs (much more slowly).
- Create and activate a Python 3.10 virtual environment:
```
python3.10 -m venv venv
source venv/bin/activate
```
- Install requirements: `pip install -r requirements`

## Getting Data
---
- Create a data directory at the root of the project: `mkdir data`
- Download an image dataset into a subdirectory with the name of the dataset
- The dataset must contain subdirectories named with the domain, then further subdirectories with the class of the contained images, such as `data/office31/amazon/calculator/`
- Modify the `DIRECTORIES` dictionary in DDC/config.py with unevaluated template strings of the directory structure of your data as needed.  See examples in the file. `%s` represents the class name, and as you can see there, some datasets like office31 contain superfluous intermediate directories ("images") that must be represented. The dictionary key must match the name you use later on the command line when running the net on this dataset.
- Here are some datasets you can use:
  - The original [Office31 dataset](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view).
  - [Office-Home dataset](https://www.hemanthdv.org/officeHomeDataset.html)

## Training and Testing
- `cd DDC`
- Model training and testing can be run via `python main.py` with various options. Run `python main.py to see all available args.
- Example: `python main.py --epochs 100 --batch_size_source 128 --batch_size_target 128 --name_source Art --name_target RealWorld --parent_dataset officehome10 --num_classes 10 --adapt_domain`
- Epochs and batch sizes are somewhat arbitrary and can generally be run as above.
- `parents_dataset` should align with a key in `DDC/config.py DIRECTORIES`
- `name_source` and `name_target` should align with domain subdirectories in the dataset. These must contain exactly the same classes as each other.
- `num_classes` must be included and set to, you guessed it, the number of classes in each domain.
- `adapt_domain` is a boolean that defaults to false and determines if a domain confusion layer and loss should be included. Run the same command once with and once without this flag in order to compare the efficacy of DDC relative to baseline performance of the un-confused net.


```
python main.py --epochs 100 --batch_size_source 128 --batch_size_target 128 --name_source amazon --name_target webcam
```

**Loss and accuracy plots**
---

Once the model is trained, you can generate plots like the ones shown above by running:

```
cd DeepCORAL/
python plot_loss_acc.py --source amazon --target webcam --no_epochs 100
```

The following is a list of the arguments the usuer can provide:

* ```--epochs``` number of training epochs
* ```--source``` name of source dataset
* ```--target``` name of source dataset



_____
## Original Readme
---
Pytorch implementation of four neural network based domain adaptation techniques: DeepCORAL, DDC, CDAN and CDAN+E. Evaluated on benchmark dataset Office31.

Paper: [Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition](https://arxiv.org/abs/2109.13420)

**Abstract**

> It has been well proved that deep networks are efficient at extracting features from a given (source) labeled dataset.
However, it is not always the case that they can generalize well to other (target) datasets which very often have a different underlying distribution. In this report, we evaluate four different domain adaptation techniques for image classification tasks: **Deep CORAL**, **Deep Domain Confusion (DDC)**, **Conditional Adversarial Domain Adaptation (CDAN)** and **CDAN with Entropy Conditioning (CDAN+E)**. The selected domain adaptation techniques are unsupervised techniques where the target dataset will not carry any labels during training phase. The experiments are conducted on the office-31 dataset.

**Results**
---

Accuracy performance on the Office31 dataset for the source and domain data distributions (with and without transfer losses).

Deep CORAL             |  DDC
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/images/DEEP_CORAL_amazon_to_webcam_test_train_accuracies.jpg)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/images/DDC_amazon_to_webcam_test_train_accuracies.jpg)

CDAN             |  CDAN+E
:-------------------------:|:-------------------------:
![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/images/CDAN_amazon_to_webcam_test_train_accuracies.png)  |  ![](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/images/CDAN_E_amazon_to_webcam_test_train_accuracies.png)

Target accuracies for all six domain shifts in Office31 dataset (amazon, webcam and dslr)

| Method         | A &#8594; W   | A &#8594; D  | W &#8594; A    | W &#8594; D  | D &#8594; A    | D &#8594; W     |
| :---:          |  :---:        |     :---:    |    :---:       |  :---:       | :---:          | :---:           |   
| No Adaptaion   | 43.1 ± 2.5    | 49.2 ± 3.7   |   35.6 ± 0.6   |  94.2 ± 3.1  | 35.4 ± 0.7     |  90.9 ± 2.4     |   
| DeepCORAL      | **49.5 ± 2.7**| 40.0 ± 3.3   | **38.3 ± 0.4** | 74.4 ± 4.3   | **38.5 ± 1.5** | **89.1 ± 4.4**  |
| DDC            | 41.7 ± 9.1    | ---          | ---            | ---          | ---            | ---             |
| CDAN           | 44.9 ± 3.3    | 49.5 ± 4.6   | 34.8 ± 2.4     | 93.3 ± 3.4   | 32.9 ± 3.4     |  88.3 ± 3.8     |
| CDAN+E         | 48.7 ± 7.5    |**53.7 ± 4.7**| 35.3 ± 2.7     |**93.6 ± 3.4**| 33.9 ± 2.2     | 87.7 ± 4.0      |



**Training and inference**
---

To train the model in your computer you must download the [**Office31**](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view) dataset and put it in your data folder. 

Execute training of a method by going to its folder (e.g. DeepCORAL):

```
cd DeepCORAL/
python main.py --epochs 100 --batch_size_source 128 --batch_size_target 128 --name_source amazon --name_target webcam
```

**Loss and accuracy plots**
---

Once the model is trained, you can generate plots like the ones shown above by running:

```
cd DeepCORAL/
python plot_loss_acc.py --source amazon --target webcam --no_epochs 10
```

The following is a list of the arguments the usuer can provide:

* ```--epochs``` number of training epochs
* ```--batch_size_source``` batch size of source data
* ```--batch_size_target``` batch size of target data
* ```--name_source``` name of source dataset
* ```--name_target``` name of source dataset
* ```--num_classes``` no. classes in dataset
* ```--load_model``` flag to load pretrained model (AlexNet by default)
* ```--adapt_domain``` bool argument to train with or without specific transfer loss

**Requirements**
---
* tqdm
* PyTorch
* matplotlib
* numpy
* pickle
* scikit-image
* torchvision

**References**
---

- [DeepCORAL](https://arxiv.org/abs/1607.01719) paper
- [DDC](https://arxiv.org/abs/1412.3474) paper
- [CDAN](https://arxiv.org/abs/1705.10667) paper
