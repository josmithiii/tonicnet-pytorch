import sys
import torch
from preprocessing.nn_dataset import bach_chorales_classic
from train.train_nn import train_TonicNet, TonicNet_lr_finder, TonicNet_sanity_test
from train.train_nn import CrossEntropyTimeDistributedLoss
from train.models import TonicNet
from eval.utils import plot_loss_acc_curves, indices_to_stream, smooth_rhythm
from eval.eval import eval_on_test_set
from eval.sample import sample_TonicNet_random

def get_device():
    """
    Get the best available device for PyTorch operations.
    Returns CUDA if available, else MPS (Metal Performance Shaders) if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Set the global device
device = get_device()
print(f"Using device: {device}")

if len(sys.argv) > 1:
    if sys.argv[1] in ['--train', '-t']:
        train_TonicNet(3000, shuffle_batches=1, train_emb_freq=1, load_path='')

    elif sys.argv[1] in ['--plot', '-p']:
        plot_loss_acc_curves()

    elif sys.argv[1] in ['--find_lr', '-lr']:
        TonicNet_lr_finder(train_emb_freq=1, load_path='')

    elif sys.argv[1] in ['--sanity_test', '-st']:
        TonicNet_sanity_test(num_batches=1, train_emb_freq=1)

    elif sys.argv[1] in ['--sample', '-s']:
        x = sample_TonicNet_random(load_path='eval/TonicNet_epoch-56_loss-0.328_acc-90.750.pt', temperature=1.0)
        indices_to_stream(x)
        smooth_rhythm()

    elif sys.argv[1] in ['--eval_nn', '-e']:
        # Create a model with the correct device setting
        model = TonicNet(nb_tags=98, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.0, device=device)
        eval_on_test_set(
            'eval/TonicNet_epoch-58_loss-0.317_acc-90.928.pt',
            model,
            CrossEntropyTimeDistributedLoss(), set='test', notes_only=True)

    elif sys.argv[1] in ['--gen_dataset', '-gd']:
        if len(sys.argv) > 2 and sys.argv[2] == '--jsf':
            for x, y, p, i, c in bach_chorales_classic('save', transpose=True, jsf_aug='all'):
                continue
        elif len(sys.argv) > 2 and sys.argv[2] == '--jsf_only':
            for x, y, p, i, c in bach_chorales_classic('save', transpose=True, jsf_aug='only'):
                continue
        else:
            for x, y, p, i, c in bach_chorales_classic('save', transpose=True):
                continue

    else:
        print("")
        print("TonicNet (Training on Ordered Notation Including Chords)")
        print("Omar Peracha, 2019")
        print("")
        print("--gen_dataset\t\t\t\t prepare dataset")
        print("--gen_dataset --jsf \t\t prepare dataset with JS Fakes data augmentation")
        print("--gen_dataset --jsf_only \t prepare dataset with JS Fake Chorales only")
        print("--train\t\t\t train model from scratch")
        print("--eval_nn\t\t evaluate pretrained model on test set")
        print("--sample\t\t sample from pretrained model")
        print("")
else:

    print("")
    print("TonicNet (Training on Ordered Notation Including Chords)")
    print("Omar Peracha, 2019")
    print("")
    print("--gen_dataset\t\t\t\t prepare dataset")
    print("--gen_dataset --jsf \t\t prepare dataset with JS Fake Chorales data augmentation")
    print("--gen_dataset --jsf_only \t prepare dataset with JS Fake Chorales only")
    print("--train\t\t\t train model from scratch")
    print("--eval_nn\t\t evaluate pretrained model on test set")
    print("--sample\t\t sample from pretrained model")
    print("")
