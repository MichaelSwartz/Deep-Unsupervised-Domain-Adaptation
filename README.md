# DDC-specifc Fork of Deep-Unsupervised-Domain-Adaptation

This is a fork of [Deep-Unsupervised-Domain-Adaptation](https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation), which provided a more-up-to-date version of the code originally developed in (DDC-transfer-learning
)[https://github.com/syorami/DDC-transfer-learning].

- Paper for that repo: [Evaluation of Deep Neural Network Domain Adaptation Techniques for Image Recognition](https://arxiv.org/abs/2109.13420)
- Paper for the original repo: [DDC](https://arxiv.org/abs/1412.3474)

## Setup 
- This project and requirements were built on a Linux machine, and is not guaranteed to work on a different operating system. It is optimized for CUDA-enabled GPUs, but can be run on typical CPUs (much more slowly).
- Create and activate a Python 3.10 virtual environment:
```
python3.10 -m venv venv
source venv/bin/activate
```
- Install requirements: `pip install -r requirements`

## Getting Data
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
- After running the model, you should move the resulting .pkl (pickle) files into the /logs directory in a similar structure to the examples contained there: `logs/{parent_dataset}/{SOURCE}_to_{target}/{EPOCHS_NUMBER_{SOURCE_BATCH_SIZE}_s_{TARGET_BATCH_SIZE}_t_batch_size`, ie `logs/office31/amazon_to_webcam/100_epochs_128_s_128_t_batch_size/`

## Loss and accuracy plots

Once the model is trained, you can generate plots like [this](/DDC/logs/office31/amazon_to_webcam_ORIGINAL/100_epochs_128_s_128_t_batch_size/amazon_to_webcam_test_train_accuracies.jpg) and [this](/DDC/logs/office31/amazon_to_webcam_ORIGINAL/100_epochs_128_s_128_t_batch_size/amazon_to_webcam_train_losses.jpg) by running, for example:

`python plot_loss_acc.py --parent_dataset officehome10 --source art  --target realworld --no_epochs 100 --include_no_adapt`

This script assumes that you have run the data with domain confusion; if you have also run it without domain confusion for the same dataset and would like to include that result for comparison, include the `include_no_adapt` flag.  
Get details on all available arguments via `python plot_loss_acc.py --help`
