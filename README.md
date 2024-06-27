## Reproduction of "Graph Neural News Recommendation with Unsupervised Preference Disentanglement"
This is a reproduction of the ACL 2020 paper “Graph Neural News Recommendation with Unsupervised Preference Disentanglement”, made for the RecSys Challenge 2024.
We have forked the authors' code and tried to use it. Because we use a different dataset, we adapted the dataloader and made a lot of changes, the new dataloader can be found in `data_loader_ebnerd.py`. We also wrote a new `main.py`. A detailed overview of our contribution can be found below.
## Our contribution
- We have added preprocessing code that loads the data from our dataset and preprocesses it. (data_loader_ebnerd.py)
- We adapted their dataloader function to make it work with the new dataset. (data_loader_ebnerd.py)
- We updated their code to work with Tensorflow 2 instead of Tensorflow 1. (model.py, train.py, aggregators.py)
- We created an environment and a job file that could be run on a compute cluster. (environment.yaml, run.job)
- We added two benchmark models to compare the model to. (main.py)