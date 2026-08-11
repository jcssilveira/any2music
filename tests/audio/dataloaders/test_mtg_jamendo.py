from itertools import islice

from torch.utils.data import DataLoader

from any2music.audio.dataloaders.mtg_jamendo import MTGJamendoDataset

def test_mtg_jamendo():
    train_dataset = MTGJamendoDataset(
        dataset_root='/home/es119256/dados/datasets/mtg-jamendo',
        mtg_clone='/home/es119256/dados/repos/passt-on-mtg-jamendo-dataset',
        split_type='train'
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for example in islice(train_loader, 10):
        print(example, '\n\n')