from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.vision.resnet import resnet50
from src.vision.part1_ppm import PPM


class PSPNet(nn.Module):
    """
    The final feature map size is 1/8 of the input image.

    Use the dilated network strategy described in
        https://arxiv.org/pdf/1511.07122.pdf

    ResNet-50 has 4 blocks, and those 4 blocks have [3, 4, 6, 3] layers,
    respectively.
    """

    def __init__(
        self,
        layers: int = 50,
        bins=(1, 2, 3, 6),
        dropout: float = 0.1,
        num_classes: int = 2,
        zoom_factor: int = 8,
        use_ppm: bool = True,
        criterion=nn.CrossEntropyLoss(ignore_index=255),
        pretrained: bool = True,
        deep_base: bool = True,
    ) -> None:
        """
        Args:
            layers: int = 50,
            bins: list of grid dimensions for PPM, e.g. (1,2,3) means to create
                (1x1), (2x2), and (3x3) grids
            dropout: float representing probability of dropping out data
            num_classes: number of classes
            zoom_factor: TODO
            use_ppm: boolean representing whether to use the Pyramid Pooling
                Module
            criterion: loss function module
            pretrained: boolean representing ...
        """
        super(PSPNet, self).__init__()
        assert layers == 50
        assert 2048 % len(bins) == 0
        assert num_classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.dropout = dropout
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.criterion = criterion

        self.layer0 = None
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None
        self.ppm = None
        self.cls = None
        self.aux = None

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        # Initialize your ResNet backbone, and set the layers                 #
        # layer0, layer1, layer2, layer3, layer4.                             #
        # Note: layer0 should be sequential                                   #
        #######################################################################

        self.deep_base = deep_base 
        resnet = resnet50(pretrained=pretrained, deep_base=self.deep_base)
        self.resnet = resnet
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.conv2,
            resnet.bn2,
            resnet.relu,
            resnet.conv3,
            resnet.bn3,
            resnet.relu,
            resnet.maxpool,
        )        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool 
        self.fc = resnet.fc

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        self.__replace_conv_with_dilated_conv()

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        # Initialize the PPM. The reduction_dim should be equal to the output #
        # number of ResNet feature maps, divided by the number of PPM bins.   #
        # Afterwards, set fea_dim to the updated feature dimension to be      #
        # passed to the classifier.                                           #
        #######################################################################

        in_dim_ppm = self.layer4[-1].conv3.out_channels 
        in_dim_ppm = 2048
        red_dim_ppm = int(2048 // len(bins))
        self.ppm = PPM(2048, red_dim_ppm, bins)
        if self.use_ppm: 
            self.fea_dim = in_dim_ppm + red_dim_ppm * len(bins)
        else:
            self.fea_dim = in_dim_ppm

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

        self.cls = self.__create_classifier(
            in_feats=self.fea_dim, out_feats=512, num_classes=num_classes)
        self.aux = self.__create_classifier(
            in_feats=1024, out_feats=256, num_classes=num_classes)

    def __replace_conv_with_dilated_conv(self):
        """
        Increase the receptive field by reducing stride and increasing
        dilation.

        In Layer3, in every `Bottleneck`, we will change the 3x3 `conv2`, we
        will replace the conv layer that had stride=2, dilation=1, and
        padding=1 with a new conv layer, that instead has stride=1, dilation=2,
        and padding=2. In the `downsample` block, we'll also need to hardcode
        the stride to 1, instead of 2.

        In Layer4, for every `Bottleneck`, we will make the same changes,
        except we'll change the dilation to 4 and padding to 4.

        Hint: you can iterate over each layer's modules using the
        .named_modules() attribute, and check the name to see if it's the one
        you want to edit. Then you can edit the dilation, padding, and stride
        attributes of the module.
        """
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        for block in self.layer3:
            block.conv2.stride = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)
            if block.downsample is not None: 
                block.downsample[0].stride = (1, 1)

        for block in self.layer4: 
            block.conv2.stride = (1, 1)
            block.conv2.dilation = (4, 4)
            block.conv2.padding = (4, 4)
            if block.downsample is not None: 
                block.downsample[0].stride = (1, 1)
        
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def __create_classifier(self, in_feats: int, out_feats: int, num_classes: int) -> nn.Module:
        """
        Implement the final PSPNet classifier over the output categories.

        Args:
            in_feats: number of channels in input feature map
            out_feats: number of filters for classifier's conv layer
            num_classes: number of output categories
        Returns:
            cls: A sequential block of 3x3 convolution, 2d Batch norm, ReLU,
                2d dropout, and a final 1x1 conv layer over the number of
                output classes. The 3x3 conv layer's padding should preserve
                the height and width of the feature map. The specified dropout
                is defined in `self.dropout`.
        """
        cls = None
        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################

        cls = nn.Sequential(
            nn.Conv2d(in_feats, out_feats, (3, 3), padding="same"), 
            nn.BatchNorm2d(out_feats),
            nn.ReLU(),
            nn.Dropout2d(self.dropout),
            nn.Conv2d(out_feats, num_classes, (1, 1))
        )

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return cls

    def forward(
        self, x: torch.Tensor, y: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the network.

        Feed the input through the network, upsample the aux output (from layer
        3) and main output (from layer4) to the ground truth resolution, and
        then compute the loss and auxiliary loss. The aux classifier should
        operate on the output of layer3. The PPM should operate on the output
        of layer4.

        Note that you can return a tensor of dummy values for the auxiliary
        loss if the model is set to inference mode. Note that nn.Module() has a
        `self.training` attribute, which is set to True depending upon whether
        the model is in in training or evaluation mode.
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#module

        Args:
            x: tensor of shape (N,C,H,W) representing batch of normalized input
                image
            y: tensor of shape (N,H,W) represnting batch of ground truth labels

        Returns:
            logits: tensor of shape (N,num_classes,H,W) representing class
                scores at each pixel
            yhat: tensor of shape (N,H,W) representing predicted labels at each
                pixel
            main_loss: loss computed on output of final classifier if y is
                provided, else return None if no ground truth is passed in
            aux_loss: loss computed on output of auxiliary classifier (from
                intermediate output) if y is provided, else return None if no
                ground truth is passed in
        """
        x_size = x.size()
        assert (x_size[2] - 1) % 8 == 0 and (x_size[3] - 1) % 8 == 0

        #######################################################################
        # TODO: YOUR CODE HERE                                                #
        #######################################################################
        h_dim, w_dim = x_size[2], x_size[3]
        
        out = self.layer0(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out3 = self.layer3(out) 
        out4 = self.layer4(out3)

        # main_output = F.interpolate(out4, (h_dim, w_dim), mode="bilinear")
        main_output = out4

        if self.use_ppm:
            out_ppm = self.ppm(main_output)
        else:
            out_ppm = main_output
        logits = self.cls(out_ppm) 
        logits = F.interpolate(logits, (h_dim, w_dim), mode="bilinear")
        yhat = torch.argmax(logits, dim=1)
        main_loss = self.criterion(logits, y) if y is not None else None

        if self.training and y is not None:
            # aux_output  = F.interpolate(out3, (h_dim, w_dim), mode="bilinear")
            aux_output = out3
            aux_logits = self.aux(aux_output)
            aux_logits = F.interpolate(aux_logits, (h_dim, w_dim), mode="bilinear")
            aux_loss = self.criterion(aux_logits, y) 
        elif not self.training: 
            aux_loss = torch.Tensor([0])
        else:
            aux_loss = None
       
        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################
        return logits, yhat, main_loss, aux_loss
