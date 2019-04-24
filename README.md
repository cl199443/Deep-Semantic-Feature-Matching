# Deep-Semantic-Feature-Matching
This is a project for solving Image Semantic Feature Matching. The main reference paper is "Deep Semantic Feature Matching" in CVPR17
and "Neural Best-Buddies: Sparse Cross-Domain Correspondence" in SIGGRAPH18.

The feature extraction module is the VGG19-net without full-connected layers. With the output of VGG19-net, a deep feature pyramid with five layers is generated, which can be found in test_vgg19.py.

In order to find pixel-level correspondence in source and target images, a coarse-to-fine matching strategy from the higher feature pyramid layers to lower layers is implemented in BPConv.py.
