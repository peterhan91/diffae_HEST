from templates import *

if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    gpus = [0, 1 ,2]
    conf = hest256_autoenc()
    train(conf, gpus=gpus)