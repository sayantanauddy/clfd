# Continual Learning from Demonstration of Robotic Skills

![Robot executing HelloWorld tasks](videos_images/helloworld_robot.gif?raw=true "Robot executing HelloWorld tasks")

![Robot executing tasks from RoboTasks ](videos_images/robotasks_pred_cropped.gif?raw=true "Robot executing tasks from RoboTasks")

Methods for teaching motion skills to robots focus on training for a single skill at a time. Robots capable of learning from demonstration can considerably benefit from the added ability to learn new movement skills without forgetting what was learned in the past. To this end, we propose an approach for continual learning from demonstration using hypernetworks and neural ordinary differential equation solvers. We empirically demonstrate the effectiveness of this approach in remembering long sequences of trajectory learning tasks without the need to store any data from past demonstrations. Our results show that hypernetworks outperform other state-of-the-art continual learning approaches for learning from demonstration. In our experiments, we use the popular LASA benchmark, and two new datasets of kinesthetic demonstrations collected with a real robot that we introduce in our [paper](https://arxiv.org/abs/2202.06843) called the *HelloWorld* and *RoboTasks* datasets. We evaluate our approach on a physical robot and demonstrate its effectiveness in learning realistic robotic tasks involving changing positions as well as orientations. We report both trajectory error metrics and continual learning metrics, and we propose two new continual learning metrics. Our code, along with the newly collected datasets, is available in this repository.

Here is a very short overview of our approach (also available on [YouTube](https://youtu.be/0gdIImIBnXc)):

https://user-images.githubusercontent.com/10401716/200596280-efc48037-02a9-42e6-88e7-e8e926c0924f.mp4

## HelloWorld Dataset
*HelloWorld* is a dataset of kinesthetic demonstrations we collected using the Franka Emika Panda robot. The *x* and *y* coordinates of the robot's end-effector were recorded while a human user guided it kinesthetically to write the 7 lower-case letters *h,e,l,o,w,r,d* one at a time on a horizontal surface. The *HelloWorld* dataset  consists of 7 tasks, each containing 8 slightly varying demonstrations of a letter. Each demonstration is a sequence of 1000 2-D points. After training on all the tasks, the objective is to make the robot write the words *hello world*. Our motivation for using this dataset is to test our approach on trajectories with loops and to show that it also works on kinesthetically recorded demonstrations using a real robot.

The data for each of the 7 tasks can be found as `.npy` files in the folder [`datasets/robot_hello_world/processed_demos`](datasets/robot_hello_world/processed_demos).

![HelloWorld_dataset](videos_images/HelloWorld_dataset.svg?raw=true "HelloWorld_dataset")

Please check the file `helloworld_dataset.ipynb` to see how to load this dataset. Code for using this dataset in the training loop can be found in the training scripts `tr_*_node.py` (e.g. `tr_hn_node.py`).

## RoboTasks Dataset
*RoboTasks* is a dataset of kinesthetic demonstrations of realistic robot tasks we collected using the Franka Emika Panda robot. Each task involves learning trajectories of the position (in 3D space) as well as the orientation (in all 3 rotation axes) of the robot's end-effector. The tasks of this dataset are: 
- Task 0 - *box opening*: the lid of a box is lifted to an open position;
- Task 1 - *bottle shelving*: a bottle in a vertical position is transferred to a horizontal position on a shelf.
- Task 2 - *plate stacking*: a plate in a vertical position is transferred to a horizontal position on an elevated platform while orienting the arm so as to avoid the blocks used for holding the plate in its initial vertical position;
- Task 3 - *pouring*: a cup full of coffee beans is taken from an elevated platform and the contents of the cup are emptied into a container.

The data for each of the 4 tasks can be found as `.npy` files in the folder [`datasets/robottasks/pos_ori`](datasets/robottasks/pos_ori). Upon loading the data (of each task), we get a numpy array of shape `[num_demos=9, trajectory_length=1000, data_dimension=7]`. A data point consists of 7 elements: `px,py,pz,qw,qx,qy,qz` (3D position followed by quaternions in the scalar first format). This represents the position and orientation of the end-effector at each point of a trajectory.

![Collecting demos for the RoboTasks dataset](videos_images/robotasks_cropped.gif?raw=true "Collecting demos for the RoboTasks dataset")

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

- A big thank you to [Hector Villeda](https://iis.uibk.ac.at/people) for helping with the data collection and robot experiments for the *HelloWorld* dataset.

- We gratefully acknowlege these openly accessible repositories which were a great help in writing the code for our experiments:

    1. [Continual Learning with Hypernetworks](https://github.com/chrhenning/hypercl)
    2. [Notebook](https://colab.research.google.com/drive/1ygdXFuih_0sLA2HosQkaVQOA9v6BMSdj?usp=sharing) containing starter code for Neural ODEs by Çağatay Yıldız
    3. [ImitationFlow](https://github.com/TheCamusean/iflow)
    4. [Fast implementations of the Frechet distance](https://github.com/joaofig/discrete-frechet)

## Citation

If you use this code or our results in your research, please cite:

```
@misc{https://doi.org/10.48550/arxiv.2202.06843,
  doi = {10.48550/ARXIV.2202.06843},
  url = {https://arxiv.org/abs/2202.06843},
  author = {Auddy, Sayantan and Hollenstein, Jakob and Saveriano, Matteo and Rodríguez-Sánchez, Antonio and Piater, Justus},
  keywords = {Robotics (cs.RO), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Continual Learning from Demonstration of Robotic Skills},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
