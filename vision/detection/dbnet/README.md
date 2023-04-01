<!--- SPDX-License-Identifier: Apache-2.0 -->

# DBnet

## Description

This is a implementation of the Differentiable Binarization. And this model perform the binarization process in a segmentation network. We directly adapt the PaddleOCR version. And we adapt MobileNet-v2 as the backbone.


## Model

|Model(nntc)                |Model(dbnet)               |H1 Mean    |
|---------------------------|---------------------------|-----------|
|[dbnet.pdmodel](inference) |[dbnet.onnx](dbnet.onnx)   |68%        |

You can use Paddle2Onnx to convert paddle-style model-storage to onnx-style.

## Dataset

We choose Task 4.1 in Incidental Scene Text Dataset icdar2015 as the benchmark. And we offer preprocessed data and groundtruth.

## References

* ["Real-time Scene Text Detection with Differentiable Binarization"](<https://arxiv.org/pdf/1911.08947.pdf>)

* [PaddleOCR](<https://github.com/PaddlePaddle/PaddleOCR>)

* [icdar2015](https://rrc.cvc.uab.es/?ch=4>)

* [Task4_1]<https://rrc.cvc.uab.es/?ch=4&com=tasks>

## License

Apache-2.0