PyTorch implementation of "Generating diverse molecular de novo structures using reinforcement learning"
=======================================================================================

This model is similar to our model used in "[Molecular De Novo Design through Deep Reinforcement
Learning](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0235-x)". It's implementation is different from the one in the paper in several ways:

     * The GRU model has an embedding layer
     * Scoring is (0, 1) rather than (-1, 1)
     * Only unique sequences are considered, ie if the same sequence is generated twice, it
       still only contributes to the loss once.
     * Sequences are penalized for being very likely. This and the point above means that the
       training is much more robust towards getting stuck in local minimum, and often very high
       values of sigma can be used if needed.
     * Prioritized experience replay is implemented. This is a little unusual for policy gradient
       since it is sensitive to how often an action is taken, but works well in some cases. (It's deactivated by
       default)

Install
-------

A Conda environment.yml is supplied with all the required libraries.

```bash
git clone https://github.com/tblaschke/reinvent
cd reinvent
conda env create -f environment.yml
conda activate reinvent
```


General usage
-------------

###Use the provided ChEMBL model
We already provide a model which is reasonably trained on ChEMBL. To get started we recommend to use this model and 
play around with the different scoring functions.

1. Run reinforce_model.py to start the reinforcement learning to generate new structures.

2. (Optional) Check out Vizor (https://github.com/tblaschke/vizor) to have a visualization for
the reinforcement learning

###Create a new model
You might be interested to train a model with your own set of compounds. To do so here is a quick list of steps you 
should follow.
 
1. Use create_model.py to preprocess a SMILES file and to build an untrained model.

2. Use train_model.py to train your model on a SMILES file.  
  * BONUS: Train_model also allows you to do some transfer learning on any model if you just 
   train an already trained Prior a second time on a small subset of compounds.

3. Run reinforce_model.py to start the reinforcement learning to generate new structures.

4. (Optional) Check out Vizor (https://github.com/tblaschke/vizor) to have a visualization for
the reinforcement learning



Write your own scoring function (the general way)
-------------------------------------------------
You can provide your own scoring function 


To write your own scoring function just drop 

Write your own scoring function in python
-----------------------------------------

To write your own scoring function just drop 