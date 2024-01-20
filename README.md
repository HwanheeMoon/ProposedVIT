##  비전 트랜스포머 성능향상을 위한 이중 구조 셀프 어텐션 A Dual-Structured Self-Attention for improving the Performance of Vision Transformers
Kwang - Yeop Lee, Hwan - Hee Moon, Tae - Ryong Park


## Abstract 
In this paper, we propose a dual-structured self-attention method that improves the lack of regional features
of the vision transformer's self-attention. Vision Transformers, which are more computationally efficient than
convolutional neural networks in object classification, object segmentation, and video image recognition, lack the
ability to extract regional features relatively. To solve this problem, many studies are conducted based on
Windows or Shift Windows, but these methods weaken the advantages of self-attention-based transformers by
increasing computational complexity using multiple levels of encoders. This paper proposes a dual-structure
self-attention using self-attention and neighborhood network to improve locality inductive bias compared to the
existing method. The neighborhood network for extracting local context information provides a much simpler
computational complexity than the window structure. CIFAR-10 and CIFAR-100 were used to compare the
performance of the proposed dual-structure self-attention transformer and the existing transformer, and the
experiment showed improvements of 0.63% and 1.57% in Top-1 accuracy, respectively.
