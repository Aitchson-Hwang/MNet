# MNet
The official code for the paper "MNet: A Multi-Scale Network for Visible Watermark Removal."

# introduction
The files related to training MNet are main.py, machines/SUNET.py (the training machine), mynets/MNet.py or MNetold.py (both are MNet model structure codes, the difference is: MNet.py sets the shared parameter UNet of the two branches to the same UNet, MNetold.py contains all the UNets of the two branches, and the shared parameter UNet will be defined twice in the two branches, but only one UNet will participate in the forward to simulate the effect of shared parameters. The two codes may have slight performance differences). Here, two .sh files are given as examples of training and testing MNet respectively (the contents of these two files need to be modified locally).
