# Continual Learning from Demonstration of Robotic Skills

![Robot executing HelloWorld tasks](videos_images/helloworld_robot.gif?raw=true "Robot executing HelloWorld tasks")

Methods for teaching motion skills to robots focus on training for a single skill at a time. Robots capable of learning from demonstration can considerably benefit from the added ability to learn new movements without forgetting past knowledge. To this end, we propose an approach for continual learning from demonstration using hypernetworks and neural ordinary differential equation solvers. We empirically demonstrate the effectiveness of our approach in remembering long sequences of trajectory learning tasks without the need to store any data from past demonstrations. Our results show that hypernetworks outperform other state-of-the-art regularization-based continual learning approaches for learning from demonstration. In our experiments, we use the popular LASA trajectory benchmark, and a new dataset of kinesthetic demonstrations that we introduce in our [paper]() called the *HelloWorld* dataset. We evaluate our approach using both trajectory error metrics and continual learning metrics, and we propose two new continual learning metrics. Our code, along with the newly collected dataset, is available in this repository.

Here is a very short overview of our approach (also available on [YouTube](https://youtu.be/cTfVfYyyeXk)):

https://user-images.githubusercontent.com/10401716/153514989-200eec00-d621-4915-a3ad-22ca3ddb4d8a.mp4


## HelloWorld Dataset
*HelloWorld* is a dataset of demonstrations we collected using the Franka Emika Panda robot. The *x* and *y* coordinates of the robot's end-effector were recorded while a human user guided it kinesthetically to write the 7 lower-case letters *h,e,l,o,w,r,d* one at a time on a horizontal surface. The *HelloWorld* dataset  consists of 7 tasks, each containing 8 slightly varying demonstrations of a letter. Each demonstration is a sequence of 1000 2-D points. After training on all the tasks, the objective is to make the robot write the words *hello world*. Our motivation for using this dataset is to test our approach on trajectories with loops and to show that it also works on kinesthetically recorded demonstrations using a real robot.

The data for each of the 7 tasks can be found as `.npy` files in the folder [`datasets/robot_hello_world/processed_demos`](datasets/robot_hello_world/processed_demos).

![HelloWorld_dataset](videos_images/HelloWorld_dataset.svg?raw=true "HelloWorld_dataset")

Please check the file `helloworld_dataset.ipynb` to see how to load this dataset. Code for using this dataset in the training loop can be found in the training scripts `tr_*_node.py` (e.g. `tr_hn_node.py`).

## Code Instructions

### Clone this repository
```
git clone https://github.com/sayantanauddy/clfd.git
```

### Create a virtual enviornment and install the dependencies

```
# Create a virtual environment, then activate it
cd <path/to/this/repository>
python -m pip install -r requirements.txt
```
Install the GPU version of `torch` if needed.

### Execute a training run

Here we show the command for training a Hypernetwork that generates a NODE:
```
# DATASET: LASA
# NODE TYPE: NODE^T (with time input)

python3 tr_hn_node.py --data_dir datasets/LASA/DataSet --num_iter 15000 --tsub 20 --replicate_num 0 --lr 0.0001 --tnet_dim 2 --tnet_arch 100,100,100 --tnet_act elu --hnet_arch 200,200,200 --task_emb_dim 256 --explicit_time 1 --beta 0.005 --data_class LASA --eval_during_train 0 --seq_file datasets/LASA/lasa_sequence_all.txt --log_dir logs_clfd/lasa_explicit_time --plot_fs 10 --figw 16.0 --figh 3.3 --task_names_path datasets/LASA/lasa_names.json --seed 200 --description tr_hn_node_LASA_t1
```
The complete set of commands for reproducing all our experiments can be found in [commands_LASA.txt](https://github.com/sayantanauddy/clfd/blob/main/commands_LASA.txt) and [commands_HelloWorld.txt](https://github.com/sayantanauddy/clfd/blob/main/commands_HelloWorld.txt).


### Reproducing results
To reproduce the results from our experiments, use the commands from [commands_LASA.txt](https://github.com/sayantanauddy/clfd/blob/main/commands_LASA.txt) and [commands_HelloWorld.txt](https://github.com/sayantanauddy/clfd/blob/main/commands_HelloWorld.txt). 

Once all the training scripts have completed executing and all log files have been generated in the folder `logs_clfd`, generate plots using the notebook `generate_plots.ipynb`.

### View Trajectories Predicted by Trained Models

First download the pretrained models and extract them to the directory `trained_models`
```
cd <path/to/this/repository>
cd trained_models
wget https://iis.uibk.ac.at/public/auddy/clfd/trained_models/trained_models.tar.xz
tar xvf trained_models.tar.xz
```

Then run the notebook `predict_traj_saved_models.ipynb` for generating trajectory predictions using the pretrained models provided by us.

## Acknowledgements

- A big thank you to [Hector Villeda](https://iis.uibk.ac.at/people) for helping with the data collection and robot experiments.

- We gratefully acknowlege these openly accessible repositories which were a great help in writing the code for our experiments:

    1. [Continual Learning with Hypernetworks](https://github.com/chrhenning/hypercl)
    2. [Notebook](https://colab.research.google.com/drive/1ygdXFuih_0sLA2HosQkaVQOA9v6BMSdj?usp=sharing) containing starter code for Neural ODEs by Çağatay Yıldız
    3. [ImitationFlow](https://github.com/TheCamusean/iflow)
    4. [Fast implementations of the Frechet distance](https://github.com/joaofig/discrete-frechet)
