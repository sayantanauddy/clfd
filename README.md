# Continual Learning from Demonstration of Robotic Skills

![Alt text](videos_images/helloworld_robot.gif?raw=true "Architectures")

Methods for teaching motion skills to robots focus on training for a single skill at a time. Robots capable of learning from demonstration can considerably benefit from the added ability to learn new movements without forgetting past knowledge. To this end, we propose an approach for continual learning from demonstration using hypernetworks and neural ordinary differential equation solvers. We empirically demonstrate the effectiveness of our approach in remembering long sequences of trajectory learning tasks without the need to store any data from past demonstrations. Our results show that hypernetworks outperform other state-of-the-art regularization-based continual learning approaches for learning from demonstration. In our experiments, we use the popular LASA trajectory benchmark, and a new dataset of kinesthetic demonstrations that we introduce in our [paper](link/to/paper) called the *HelloWorld* dataset. We evaluate our approach using both trajectory error metrics and continual learning metrics, and we propose two new continual learning metrics. Our code, along with the newly collected dataset, is available in this repository.

Here is a very short overview of our approach:
https://user-images.githubusercontent.com/10401716/153514989-200eec00-d621-4915-a3ad-22ca3ddb4d8a.mp4

## HelloWorld Dataset
*HelloWorld* is a dataset of demonstrations we collected using the Franka Emika Panda robot. The $x$ and $y$ coordinates of the robot's end-effector were recorded while a human user guided it kinesthetically to write the 7 lower-case letters *h,e,l,o,w,r,d* one at a time on a horizontal surface. The *HelloWorld* dataset  consists of 7 tasks, each containing 8 slightly varying demonstrations of a letter. Each demonstration is a sequence of 1000 2-D points. After training on all the tasks, the objective is to make the robot write the words *hello world*, as shown below. Our motivation for using this dataset is to test our approach on complicated trajectories (with loops) and to show that it also works on kinesthetically recorded demonstrations using a real robot.

The data for each of the 7 tasks can be found as `.npy` files in the folder [`datasets/robot_hello_world/processed_demos`](datasets/robot_hello_world/processed_demos).

## Setup Insrtuctions

1. Clone this repository
```
git clone https://github.com/sayantanauddy/clfd.git
```

2. Create a virtual enviornment and install the dependencies

```
# Create a virtual environment, then activate it
cd <path/to/this/repository>
python -m pip install -r requirements.txt
```

3. Train HN (hypernetwork+NODE)
```

```

## View Trajectories Predicted by Trained Models

## Acknowledgements

- A big thank you to [Hector Villeda](https://iis.uibk.ac.at/people) for helping with the data collection and robot experiments.

- We gratefully acknowlege these openly accessible repositories which were a great help in writing the code for our experiments:

    1. [Continual Learning with Hypernetworks](https://github.com/chrhenning/hypercl)
    2. [Notebook](https://colab.research.google.com/drive/1ygdXFuih_0sLA2HosQkaVQOA9v6BMSdj?usp=sharing) containing starter code for Neural ODEs by Çağatay Yıldız
    3. [ImitationFlow](https://github.com/TheCamusean/iflow)
    4. [Fast implementations of the Frechet distance](https://github.com/joaofig/discrete-frechet)
