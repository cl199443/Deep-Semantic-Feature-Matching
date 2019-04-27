# Deep-Semantic-Feature-Matching
This is a project for solving Image Semantic Feature Matching. The main reference paper is "Deep Semantic Feature Matching"[1] in CVPR17
and "Neural Best-Buddies: Sparse Cross-Domain Correspondence"[2] in SIGGRAPH18.

The feature extraction module is the VGG19-net without full-connected layers. With the output of VGG19-net, a deep feature pyramid with five layers is generated, which can be found in test_vgg19.py.

In order to find pixel-level correspondence in source and target images, a coarse-to-fine matching strategy from the higher feature pyramid layers to lower layers is implemented in BPConv.py.

In order to increase the time efficiency, I have tried to make some coding tricks in the BPConvCopy.py.

[1] N. Ufer, B. Ommer. Deep semantic feature matching. In CVPR ,2017.

[2] K. Aberman, J. Liao, M. Shi, D. Lischinski, B. Chen, and D. Cohen. Neural Best-Buddies: Sparse Cross-Domain Correspondence. ACM Transactions on Graphics, 37(4):69, 2018.
