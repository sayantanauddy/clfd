# Continual Imitation Learning

Tested on 5 tasks for the LASA dataset. Each row is for training on a new task, and each column corresponds to evaluation on a past/current task. For example, the last row and second column shows the predicted trajectory when the network is trained on the 5th task, and afterwards evaluated on the 2nd task.

![LASA 5 tasks](plots/LASA_5_tasks.png)

## TODOs
1. For training a NODE, the time vector needs to be the same for all sequences. In LASA, the time vectors for different vectors is different. Using the time vector from the first sequence for all sequences for now.

## Repositories to thank

1. [ImitationFlow](https://github.com/TheCamusean/iflow)
2. [Notebook](https://colab.research.google.com/drive/1ygdXFuih_0sLA2HosQkaVQOA9v6BMSdj?usp=sharing) containing starter code for Neural ODEs by Çağatay Yıldız
3. [Continual Learning with Hypernetworks](https://github.com/chrhenning/hypercl)
4. [Fast implementations of the Frechet distance](https://github.com/joaofig/discrete-frechet)
