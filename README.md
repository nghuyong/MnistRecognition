# MnistRecognition
MNIST handwritten digit recognition system (Artificial Intelligence final project)

## Supervised Learning

|model|Accuracy|
|:---:|:---:|
|two layer full connected network|97.546|
|two layer cnn and a full connected layer|99.219|

Use two layer full connected network to study accuracy in different training samples

![](./images/accuracy_vs_samples.png)

You can see that 81% is achieved with only 250 samples and 90% with 2000. You need 27x the data to go from 90% to 97.5% with the simple network.

## Semi-Supervised Learning

What about a world where you only have **2000 labeled samples** but **have another 53000 unlabeled samples**?
Can we somehow learn something in an unsupervised way from all 55000 digits that might help us to better than our 90% baseline?
This is the goal of semi-supervised learning.

### AutoEncoder
First we learn an AutoEncoder,that encoder the input image to a embedding and then
decode the embedding to reconstruct the image.

The following gif is the AutoEncoder learning process,reconstruct image at different stages of learning:

![](./images/reconstructed.gif)

we can also directly randomly generate embedding vector and  construct image

![](./images/digit.png)

Below is a scatter plot of the 10,000 training samples from MNIST embedded in a 2-D space.

![](./images/ae-cluster.png)

### Help Supervised Learning

We send the embedding(learned from autoencoder) of the image into a fully connected neural network.

|model|Accuracy|
|:---:|:---:|
|2000 labeled samples,supervised|88.030|
|2000 labeled samples,53000 unlabeled samples,semi-supervised|92.176|

Reference:
- [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)

## Reinforcement Learning
Neural Architecture Search (NAS) with Reinforcement Learning is a method for finding good neural networks architecture.
For this part, we will try to find optimal architecture for Convolutional Neural Network (CNN) which recognizes handwritten digits.

For a signal layer network,we search the hidden layer size from 10 to 800 and step is 20.

The result is the following:

|HiddenSize|Accuracy|Reward|totalReward|
|:---:|:---:|:---:|:---:|
160|0.9781|0.1|0.1
450|0.9816|0.0035|0.1035
40|0.9687|-0.0101|0.0934
90|0.9765|-0.00028|0.09312
220|0.9797|0.00298|0.0961
30|0.9631|-0.01422|0.08188
580|0.9817|0.00722|0.0891
470|0.982|0.00608|0.09518
410|0.9813|0.00416|0.09934
160|0.9779|-7e-05|0.09928
**710**|**0.983**|**0.00504**|0.10432
420|0.9813|0.00234|0.10666
50|0.9689|-0.01053|0.09613
610|0.9821|0.00477|0.1009
160|0.9798|0.00152|0.10242
280|0.9805|0.00192|0.10434
140|0.9795|0.00053|0.10487
740|0.9837|0.00463|0.1095
380|0.9814|0.0014|0.1109
360|0.9816|0.00132|0.11222
530|0.9831|0.00256|**0.11477**
160|0.9804|-0.00065|0.11412
100|0.9769|-0.00402|0.1101
660|0.9829|0.00278|0.11288
600|0.9824|0.00172|0.1146
60|0.9747|-0.00632|0.10828
230|0.9798|4e-05|0.10832
190|0.9809|0.00114|0.10946

For a signal layer network,we search the two layers cnn parameters.

The result is the following:

|CNN_1_kernel|CNN_1_kernel|CNN_2_kernel|CNN_2_kernel|Accuracy|Reward|totalReward|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3|32|2|16|0.9904|0.1|0.1
**3**|**64**|**5**|**64**|**0.9929**|**0.0025**|0.1025
5|64|3|32|0.9923|0.0014|0.1039
5|64|4|32|0.9923|0.00112|0.10502
3|64|4|16|0.9917|0.0003|0.10532
4|64|5|16|0.9915|4e-05|**0.10535**

Overview

|Model|Accuracy|
|:---:|:---|
|Supervised learning fc|97.546|
|Using RL,Supervised learning fc|98.300|
|Supervised learning cnn fc|99.219|
|Using RL,Supervised learning cnn fc|99.240|

Reference:
- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578.pdf)
- [neural-architecture-search](https://github.com/titu1994/neural-architecture-search)

## Evolutionary Computation
Here, we try to improve upon the brute force method by applying a genetic algorithm to evolve a network with the goal of achieving optimal hyperparameters in a fraction the time of a brute force search.

### How much faster?
Let’s say it takes five minutes to train and evaluate a network on your dataset. And let’s say we have four parameters with five possible settings each. To try them all would take (5**4) * 5 minutes, or 3,125 minutes, or about 52 hours.

Now let’s say we use a genetic algorithm to evolve 10 generations with a population of 20 (more on what this means below), with a plan to keep the top 25% plus a few more, so ~8 per generation. This means that in our first generation we score 20 networks (20 * 5 = 100 minutes). Every generation after that only requires around 12 runs, since we don’t have the score the ones we keep. That’s 100 + (9 generations * 5 minutes * 12 networks) = 640 minutes, or 11 hours.

We’ve just reduced our parameter tuning time by almost 80%! That is, assuming it finds the best parameters…

### How do genetic algorithms work?

At its core, a genetic algorithm…

1. Creates a population of (randomly generated) members
2. Scores each member of the population based on some goal. This score is called a fitness function.
3. Selects and breeds the best members of the population to produce more like them
4. Mutates some members randomly to attempt to find even better candidates
5. Kills off the rest (survival of the fittest and all), and
6. Repeats from step 2. Each iteration through these steps is called a generation.
7. Repeat this process enough times and you should be left with the very best possible members of a population. Sounds like a lot evolution, right? Same deal.

### Applying genetic algorithms to Neural Networks
We’ll attempt to evolve a fully connected network (MLP). Our goal is to find the best parameters for an image classification task.

We’ll tune four parameters:

- Number of layers (or the network depth)
- Neurons per layer (or the network width)
- Dense layer activation function
- Network optimizer

The steps we’ll take to evolve the network, similar to those described above, are:

1. Initialize N random networks to create our population.
2. Score each network. This takes some time: We have to train the weights of each network and then see how well it performs at classifying the test set. Since this will be an image classification task, we’ll use classification accuracy as our fitness function.
3. Sort all the networks in our population by score (accuracy). We’ll keep some percentage of the top networks to become part of the next generation and to breed children.
4. We’ll also randomly keep a few of the non-top networks. This helps find potentially lucky combinations between worse-performers and top performers, and also helps keep us from getting stuck in a local maximum.
5. Now that we’ve decided which networks to keep, we randomly mutate some of the parameters on some of the networks.
6. Here comes the fun part: Let’s say we started with a population of 20 networks, we kept the top 25% (5 nets), randomly kept 3 more loser networks, and mutated a few of them. We let the other 12 networks die. In an effort to keep our population at 20 networks, we need to fill 12 open spots. It’s time to breed!

#### Breeding
Breeding is where we take two members of a population and generate one or more child, where that child represents a combination of its parents.

In our neural network case, each child is a combination of a random assortment of parameters from its parents. For instance, one child might have the same number of layers as its mother and the rest of its parameters from its father. A second child of the same parents may have the opposite. You can see how this mirrors real-world biology and how it can lead to an optimized network quickly.

### Result

FC (origin Accuracy 97.546)

||Brute search | GA search|
|:---:|:---:|:---:|
|Time|64.1h|4.3h|
|FC Accuracy|98.130|98.290|


CNN+FC (origin Accuracy 99.219)

||Brute search | GA search|
|:---:|:---:|:---:|
|Time|72.3h|7.6h|
|CNN Accuracy|99.238|99.239|
Reference:
- [Let’s evolve a neural network with a genetic algorithm](https://blog.coast.ai/lets-evolve-a-neural-network-with-a-genetic-algorithm-code-included-8809bece164)
- [EDEN: Evolutionary Deep Networks for Efficient
Machine Learning](https://arxiv.org/pdf/1709.09161.pdf)
- [Picture_Evolution](https://github.com/ncblair/Picture_Evolution)