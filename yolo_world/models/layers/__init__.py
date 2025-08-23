# Copyright (c) Tencent Inc. All rights reserved.
# Basic brick modules for PAFPN based on CSPLayers

from .yolo_bricks import (
    CSPLayerWithTwoConv,
    MaxSigmoidAttnBlock,
    MaxSigmoidCSPLayerWithTwoConv,
    ImagePoolingAttentionModule,
    RepConvMaxSigmoidCSPLayerWithTwoConv,
    RepMaxSigmoidCSPLayerWithTwoConv
)

from .our_bricks import (
    SoftmaxSigmoidAttnBlock,
    SoftmaxSigmoidCSPLayerWithTwoConv,
)

__all__ = ['CSPLayerWithTwoConv',
           'MaxSigmoidAttnBlock',
           'SoftmaxSigmoidAttnBlock',
           'SoftmaxSigmoidCSPLayerWithTwoConv',
           'MaxSigmoidCSPLayerWithTwoConv',
           'RepConvMaxSigmoidCSPLayerWithTwoConv',
           'RepMaxSigmoidCSPLayerWithTwoConv',
           'ImagePoolingAttentionModule']
