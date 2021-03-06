import argparse

#reference: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws_demo.ipynb
def get_train_args():
    parser = argparse.ArgumentParser()
    # model training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
                        
    parser.add_argument('--depth', type=int, default=12, help='depth')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
    parser.add_argument('--num_heads', type=int, default=4, help='num_heads')
    parser.add_argument('--patch_num', type=int, default=16, help='patch_num')

    # where dataset will be stored
    parser.add_argument("--path", type=str, default="data/speech_commands/")

    # 35 keywords + silence + unknown
    parser.add_argument("--num-classes", type=int, default=37)
   
    # mel spectrogram parameters
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)

    # 16-bit fp model to reduce the size
    parser.add_argument("--precision", default=16)
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--num-workers", type=int, default=48)

    parser.add_argument("--no-wandb", default=False, action='store_true')

    args = parser.parse_args("")
    return args