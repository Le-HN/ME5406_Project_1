# ME5406 Project 1
## The Froze Lake Problem and Variations
### Problem Statement
This project is aimed to help the robot to find a safe way to pick up a frisbee in a frozen lake
with several holes covered by patches of very thin ice using basic Reinforcement Learning
(RL) method.

There are 2 maps which have 4∗4 and 10∗10 grid respectively. The proportion between the
number of holes and the number of states is maintained at 0.25. The start point and the
destination are all set in the top left corner and the bottom right corner of both maps. All
actions the robot can do contain moving up, down, left and right. When the robot fall into a
hole, it will receive a -1 reward. If the robot get the frisbee, it will receive a +1 reward.

In this project, three basic RL methods will be used on both maps, which are first-visit
Monte Carlo control without exploring starts, SARSA with an epsilon-greedy behavior policy and
Q-learning with an epsilon-greedy behavior policy.

![image](https://github.com/Le-HN/ME5406_Project_1/blob/main/README_images/ILLUSTRATION.png)

### Performances of the algorithms


### Steps of running the code:

**1. Environment settings:**

    ale-py==0.7.1
    certifi==2016.2.28
    cloudpickle==2.0.0
    cycler==0.10.0
    gym==0.20.0
    importlib-metadata==4.8.1
    importlib-resources==5.2.2
    kiwisolver==1.3.1
    matplotlib==3.3.4
    numpy==1.19.5
    opencv-python==4.5.3.56
    Pillow==8.3.2
    pyglet==1.5.21
    pyparsing==2.4.7
    python-dateutil==2.8.2
    scipy==1.5.4
    six==1.16.0
    typing-extensions==3.10.0.2
    wincertstore==0.2
    zipp==3.6.0

You can use the requirements file in the folder to install the environment:

**Format:**

pip install -r [FILE_PATH]/requirements.txt

You should replace the [FILE_PATH] with the real file path on your computer.

**Example:**

pip install -r E:/ME5406_Project_1/requirements.txt

**2. Run the main.py**

You need to define the RL algorithm type (MONTE_CARLO/SARSA/Q_LEARNING), map size (4/10), and training iteration then run the main.py through command line.

**Format:**

python [FILE_PATH]/main.py -t <algorithm_type> -s <map_size> -i <training_iteration>

You should replace the [FILE_PATH] with the real file path on your computer.

**Example:**

python E:/ME5406_Project_1/main.py -t Q_LEARNING -s 4 -i 500

It means using Q_LEARNING in 4*4 map at training iteration 500.

After training, the plot of average reward, average q value will be shown.

If the training is successful, The route will also be shown.