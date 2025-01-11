# MNet
The official code for the paper "MNet: A Multi-Scale Network for Visible Watermark Removal." (Published in Neural Networks 183(2025) )

# Code Introduction
The files related to training MNet are main.py, machines/SUNET.py (the training machine), mynets/MNet.py or MNetold.py (both are MNet model structure codes, the difference is: MNet.py sets the shared parameter UNet of the two branches to the same UNet, MNetold.py contains all the UNets of the two branches, and the shared parameter UNet will be defined twice in the two branches, but only one UNet will participate in the forward to simulate the effect of shared parameters. The two codes may have slight performance differences). Here, two .sh files are given as examples of training and testing MNet respectively (the contents of these two files need to be modified locally).

# Checkpoints
We provide MNet's ckpts on various datasets (trained using MNetold.py): [MNet](https://drive.google.com/drive/folders/1Nvg9K7t90PZdCE-kPtXua7U5i418W99s?usp=drive_link)

# Datasets
The datasets used in this paper are from [SplitNet](https://github.com/vinthony/deep-blind-watermark-removal)

# Citation
```bibtex
@article{huang2025mnet,
  title={MNet: A multi-scale network for visible watermark removal},
  author={Huang, Wenhong and Dai, Yunshu and Fei, Jianwei and Huang, Fangjun},
  journal={Neural Networks},
  volume={183},
  pages={106961},
  year={2025},
  publisher={Elsevier}
}
```

# Acknowledgement
This code is mainly based on the previous works
[SplitNet](https://github.com/vinthony/deep-blind-watermark-removal),
[SLBR](https://github.com/bcmi/SLBR-Visible-Watermark-Removal),
[DENet](https://github.com/lianchengmingjue/DENet)
