import os

######################################## Pytorch lightning ########################################################

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
seed_everything(13)

######################################## Model and Dataset ########################################################

from Network_Real_Burst import Burstormer
from datasets.burstsr_dataset import BurstSRDataset
from torch.utils.data.dataloader import DataLoader

##################################################################################################################

log_dir = './logs/Track_2/'

pretrained_burstormer_path = ''

class Args:
    def __init__(self):
        self.path = "./data/burstsr_dataset"
        self.checkpoint_path = log_dir + "saved_model"
        self.result_dir = log_dir + "results"
        self.batch_size = 1
        # self.num_epochs = 50
        self.lr = 1e-4
        self.burst_size = 8
        self.NUM_WORKERS = 4
args = Args()

######################################### Data loader ######################################################

def load_data():
    train_loader = DataLoader(
        dataset=BurstSRDataset(args.path, burst_size = 8, split='train') , batch_size=args.batch_size, num_workers=4, shuffle=True
    )    

    test_loader = DataLoader(
        dataset=BurstSRDataset(args.path, burst_size = 8, split='val'), batch_size=args.batch_size, num_workers=4, shuffle=False
    )
    return train_loader, test_loader

######################################### Logger ####################################################

logger = TensorBoardLogger(log_dir, name="real_burst_sr")


checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    save_last=True,
    dirpath=args.checkpoint_path,
    filename='model_{epoch:02d}-{val_loss:.2f}'
)

######################################### Load Burstormer ####################################################

if os.path.exists(args.checkpoint_path):
    print("Loading from the last checkpoint.")
    last_model_path = checkpoint_callback.last_model_path
    model = Burstormer.load_from_checkpoint(last_model_path)
elif os.path.exists(pretrained_burstormer_path): # path to pretrained Burstormer on synthetic dataset
    print("Loading the pretrained burstormer on synthetic dataset...")
    model = Burstormer.load_from_checkpoint(pretrained_burstormer_path)
else : 
    print("training from scratch")
    model = Burstormer()


######################################### Training #######################################################

train_loader, test_loader = load_data()

trainer = Trainer(  
                limit_train_batches=4,
                limit_val_batches=2,
                max_epochs=5,
                precision=32, # 16 = half, 32 = float, 64 = double
                gradient_clip_val=0.01,
                callbacks=[checkpoint_callback],
                logger=logger,
                benchmark=True, # speeds up training by finding optimal algos
            )

if(__name__ == '__main__'):
    trainer.fit(model, train_dataloaders= train_loader, val_dataloaders= test_loader)
