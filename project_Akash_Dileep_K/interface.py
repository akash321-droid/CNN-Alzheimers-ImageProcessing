# replace MyCustomModel with the name of your model
from model import CombinedCNN_small as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import run_training as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import analyze_alzheimers_mri as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import AlzheimersDataset as TheDataset

# change unicornLoader to your custom dataloader
from dataset import create_alzheimer_data_loaders as the_dataloader