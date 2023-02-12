# Part 1: dvc - Data version control

## Installation 
1. pip install dvc 
2. pip install dvc-gdrive

## Initialization
1. git init
2. dvc init


## Store your data on the Google Drive 
![](../Downloads/330732868_557289109672356_8788318334601107559_n.png)

1. dvc add data/MNIST #Replace MNIST with the name of your data folder
2. dvc add data/.gitignore data/MNIST.dvc
3. git commit -m "Add data"
4. dvc remote add -d storage drive://1XEbi8csizyjywDsHB-CPsIpeWd3IMNFL #Replace your google drive URL (see image above) 
5. git commit .dvc/config -m "Configure remote storage"
6. dvc push

## Update the data on Google Drive
1. dvc add data/MNIST
2. git add data/MNIST.dvc
3. git commit -m "Modify data"
4. dvc push


## Get the latest version of the data
1. dvc pull

## Switch back to the older version
1. git log --oneline
2. git checkout "id" data/MNIST.dvc
3. dvc checkout

# Part 2: Wandb

## Installation
1. pip install wandb

## Login

1. wandb.login()

## Initialize with hyperparameters
1. wandb.init(project="cmput469-demo", config=args)

## Log the training loss
1. wandb.log({"Train/Loss": avg_loss}, step=epoch)

## Keep track the model
1. wandb.watch(model, log_graph=True)

## Hyperparameter tuning (grid search)


# Reference
1. [Wandb Tutorial](https://wandb.ai/quickstart )
2. [dvc Tutorial](https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial)
3. [Developer0115 Blog](https://dvelopery0115.github.io/2021/08/01/Introduction_to_W&B.html)