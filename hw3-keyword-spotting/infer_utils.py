import argparse

#reference: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2022/supervised/python/kws-infer.py
def get_infer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=512)
    #parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="https://github.com/samsonharveyy/coe197-homeworks/releases/download/v1.0/kws-best-acc.pt")
    parser.add_argument("--gui", default=True, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument('--patch_num', type=int, default=16, help='patch_num')
    args = parser.parse_args()
    return args