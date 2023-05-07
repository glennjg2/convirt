
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=6464)
    parser.add_argument('-p', '--dropout-p', type=float, default=0.2)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('-lrf', '--lr-factor', type=float, default=0.5)
    parser.add_argument('-wd', '--weight-decay', type=float, default=1e-6)
    parser.add_argument('-lrp', '--lr-patience', type=int, default=5)
    parser.add_argument('-fc', '--fc-features', type=int, default=1024)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-st', '--shuffle-train', action='store_true', default=True)
    parser.add_argument('-wb', '--whole-batches', action='store_true', default=True)
    parser.add_argument('-e', '--max-epochs', type=int, default=200)
    parser.add_argument('-log', '--log-every-n-steps', type=int, default=50)
    parser.add_argument('-stop', '--early-stop-patience', type=int, default=10)
    parser.add_argument('-k', '--save-top-k', type=int, default=1)
    parser.add_argument('-f', '--checkpoint-filename', type=str, default='best-{epoch:03d}-{val_auc:.4f}')
    parser.add_argument('-train', '--train-subset', type=(float or int))
    parser.add_argument('-val', '--val-subset', type=(float or int))
    parser.add_argument('-test', '--test-subset', type=(float or int))
    parser.add_argument('-fast', '--fast-dev-run', action='store_true', default=False)
    parser.add_argument('--limit-train-batches', type=int)
    parser.add_argument('--limit-val-batches', type=int)
    parser.add_argument('--limit-test-batches', type=int)
    parser.add_argument('-d', '--encoding-size', type=int, default=512)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--loss-w-lambda', type=float, default=0.75)
    parser.add_argument('--load-checkpoint', type=str)
    parser.add_argument('--convirt-checkpoint-path', type=str)
    parser.add_argument('--train-target', type=str, default='imagenet', choices=['imagenet', 'convirt'])
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    subs = parser.add_subparsers(required=True, dest='model')
    subs.add_parser('rsna')
    subs.add_parser('covidx')
    subs.add_parser('convirt')

    return parser.parse_args()