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

class Args:
    def __init__(self):
        self.path = "./data/burstsr_dataset"
        self.checkpoint_path = log_dir + "saved_model"
        self.result_dir = log_dir + "results"
        self.batch_size = 1
        # self.num_epochs = 50
        self.lr = 3e-4
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

######################################### Load Burstormer ####################################################

# model = Burstormer()
# model.cuda()

# if not os.path.exists(args.checkpoint_path):
#     os.makedirs(args.checkpoint_path, exist_ok=True) 

# if os.path.exists(args.checkpoint_path):
#     best_model_path = checkpoint_callback.best_model_path
#     model = Burstormer.load_from_checkpoint(args.checkpoint_path)
# else : 
#     model = Burstormer()


######################################### Logger ####################################################

logger = TensorBoardLogger(log_dir, name="real_burst_sr")

######################################### Training #######################################################

train_loader, test_loader = load_data()

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=3,
    save_last=True,
    dirpath=args.checkpoint_path,
    filename='model_{epoch:02d}-{val_loss:.2f}'
)

# if os.path.exists(args.checkpoint_path):
#     last_model_path = checkpoint_callback.last_model_path
#     model = Burstormer.load_from_checkpoint(last_model_path)
# else : 
model = Burstormer()

# checkpoint_callback.

trainer = Trainer(  
                limit_train_batches=4,
                limit_val_batches=2,
                max_epochs=5,
                precision=32, # 16 = half, 32 = float, 64 = double
                gradient_clip_val=0.01,
                callbacks=[checkpoint_callback],
                logger=logger,
                # benchmark=True, # speeds up training by finding optimal algos
                # deterministic=True,
                # val_check_interval=0.25,
            )

if(__name__ == '__main__'):
    # if os.path.exists(args.checkpoint_path):
    #     last_model_path = checkpoint_callback.last_model_path
    trainer.fit(model, train_dataloaders= train_loader, val_dataloaders= test_loader)

# trainer.save_checkpoint(args.checkpoint_path, weights_only=True)