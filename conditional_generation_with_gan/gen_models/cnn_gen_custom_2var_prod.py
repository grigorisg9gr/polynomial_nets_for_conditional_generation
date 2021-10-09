import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from source.links.sn_linear import SNLinear
from source.links.sn_convolution_2d import SNConvolution2D
from source.links.instance_normalization import InstanceNormalization
from functools import partial

from source.miscs.random_samples import sample_continuous, sample_categorical


def return_norm(norm):
    if isinstance(norm, str):
        return norm
    if norm == 0:
        return 'batch'
    elif norm == 1:
        return 'instance'
    elif norm == 2:
        return 'cinstance'


def _upsample(x):
    h, w = x.shape[2:]
    return F.unpooling_2d(x, 2, outsize=(h * 2, w * 2))


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.average_pooling_2d(x, 2)


class ProdPoly2InVarConvGenerator(chainer.Chain):
    def __init__(self, layer_d, use_bn=False, sn=False, out_ch=2, mult_lat=True,
                 distribution='uniform', ksizes=None, strides=None, paddings=None, 
                 activ=None, bottom_width=1, use_localz=True, n_classes=0, 
                 use_out_act=False, channels=None, power_poly=[4, 3], 
                 derivfc=0, activ_prod=0, use_bias=True, dim_z=128, train=True,
                 add_h_injection=False, add_h_poly=False, norm_after_poly=False,
                 normalize_preinject=False, use_act_zgl=False, allow_conv=False,
                 repeat_poly=1, share_first=False, type_norm='batch', typen_poly='batch',
                 skip_rep=False, order_out_poly=None, use_activ=False, thresh_skip=None,
                 use_sec_recurs=False, sign_srec=-1, use_add_noise=False, glorot_scale=1.0,
                 dim_c=None, use_locz_c=False, power_poly_c=None, channels_c=None,
                 use_order_out_z=False, order_bef_out_poly=None):
        """
        Initializes the product polynomial generator with 2 inputs.
        :param layer_d: List of all layers' inputs/outputs channels (including 
                        the input to the last output).
        :param use_bn: Bool, whether to use batch normalization.
        :param sn: Bool, if True use spectral normalization.
        :param out_ch: int, Output channels of the generator.
        :param ksizes: List of the kernel sizes per layer.
        :param strides: List of the stride per layer.
        :param paddings: List of the padding per layer (int).
        :param activ: str or None: The activation to use.
        :param bottom_width: int, spatial resolution to reshape the init noise.
        :param use_localz: bool, if True, then use 'local' transformations (affine)
               to transform from the original z to the current shape.
        :param use_out_act: bool, if True, it uses a tanh as output activation.
        :param repeat_poly: int or list: List of how many times to repeat each 
               polynomial (original with FC layers, the last is NOT repeated).
        :param share_first: bool. If True, it shares the first FC layer in the
               FC polynomials. It assumes the respective FC layers are of the same size.
        :param type_norm: str. The type of normalization to be used, i.e. 'batch' for
               BN, 'instance' for instance norm.
        :param typen_poly: str. The type of normalization to be used ONLY 
               for the FC layers. 
        :param skip_rep: bool. If True, it uses a skip connection before the layer to 
               the next, i.e. in practice it directly skips the convolution/fc plus hadamard 
               product of this layer. See derivation 3.
        :param order_out_poly: int or None. If int, a polynomial of the order is used
               in the output, i.e. before the final convolution.
        :param use_activ: bool. If True, use activations in the main deconvolutional part.
        :param thresh_skip: None, int or list. It represents the probability of skipping the
               hadamard product in the deconvolution part (i.e. main polynomial). This is per
               layer; e.g. if 0.3, this means that approx. 30% of the time this hadamard will
               be skipped. It should have values in the [0, 1]. In inference, it should be
               deactivated.
        :param use_sec_recurs: bool, default False. If True, it uses the second recursive term, 
               i.e. x_(n+1)=f(x_n, x_(n-1)). This is used only in the derivation 3 setting. Also, 
               for practical reasons, use only for the FC layers when their order > 3.
        :param use_add_noise: bool, default False. If True, it draws additive Gaussian noise (scalar)
               after each convolutional layer.
        :param dim_c: None or int. The dimensionality of the second input variable. If None,
                then assume the n_classes to be the second input.
        :param use_locz_c: bool. If True, then also inject the c in the upsampling
                polynomial.
        :param use_order_out_z: bool. If True, convert the order_out_poly into a 2-variable  polynomial.
        :param order_bef_out_poly: int or None. If int, a polynomial of the order is used
               before the output polynomial, i.e. see order_out_poly above.
        """
        super(ProdPoly2InVarConvGenerator, self).__init__()
        self.n_l = n_l = len(layer_d) - 1
        w = chainer.initializers.GlorotUniform(glorot_scale)
        if sn:
            Conv = SNConvolution2D
            Linear = SNLinear
            raise RuntimeError('Not implemented!')
        else:
            Conv = L.Deconvolution2D
            Linear = L.Linear
        Conv0 = L.Convolution2D
        # # initialize args not provided.
        if ksizes is None:
            ksizes = [4] * n_l
        if strides is None:
            strides = [2] * n_l
        if paddings is None:
            paddings = [1] * n_l
        # # the length of the ksizes and the strides is only for the conv layers, hence
        # # it should be one number short of the layer_d.
        assert len(ksizes) == len(strides) == len(paddings) == n_l
        # # save in self, several useful properties.
        self.use_bn = use_bn
        self.n_channels = layer_d
        self.mult_lat = mult_lat
        self.distribution = distribution
        # # Define dim_z.
        assert isinstance(dim_z, int)
        self.dim_z = dim_z
        self.train = train
        # # bottom_width: The default starting spatial resolution of the convolutions.
        self.bottom_width = bottom_width
        self.use_localz = use_localz
        self.n_classes = n_classes
        if activ is not None:
            activ = getattr(F, activ)
        elif use_activ:
            activ = F.relu
        else:
            activ = lambda x: x
        self.activ = self.activation = activ
        self.out_act = F.tanh if use_out_act else lambda x: x
        # # Set the add_one to True out of consistency with the other 
        # # implementation (resnet-based).
        self.add_one = 1
        self.add_h_injection = add_h_injection
        self.normalize_preinject = normalize_preinject
        self.use_bias = use_bias
        self.add_h_poly = add_h_poly
        self.power_poly = power_poly
        if channels is None:
            # # Initialize in the dimension of z.
            channels = [dim_z] * len(self.power_poly)
        self.channels = channels
        if isinstance(activ_prod, int):
            activ_prod = [activ_prod] * len(self.power_poly)
        self.activ_prod = activ_prod
        if isinstance(repeat_poly, int):
            repeat_poly = [repeat_poly] * len(self.power_poly)
        self.repeat_poly = repeat_poly
        assert len(self.power_poly) == len(self.activ_prod) == len(self.channels)
        assert len(self.power_poly) == len(self.repeat_poly)
        # # If True, it compensates for python open interval in the end of range. 
        self.derivfc = derivfc
        self.norm_after_poly = norm_after_poly
        # # whether to use an activation in the global transformation.
        self.use_act_zgl = use_act_zgl and use_activ
        self.share_first = share_first
        self.type_norm = return_norm(type_norm)
        self.typen_poly = return_norm(typen_poly)
        self.skip_rep = skip_rep
        # # optionally adding one 'output polynomial', before making the final output convolution.
        self.order_out_poly = order_out_poly
        # # set the threshold for skipping the injection in the deconvolution part.
        if isinstance(thresh_skip, float) and self.train:
            self.thresh_skip = [0] + [thresh_skip] * (n_l - 1)
            print('[Gen] Skip probabilities: {}.'.format(self.thresh_skip))
        else:
            self.thresh_skip = thresh_skip
        self.use_sec_recurs = use_sec_recurs and skip_rep
        self.sign_srec = sign_srec
        self.use_add_noise = use_add_noise
        if dim_c is None:
            dim_c = n_classes
            assert n_classes > 0
        self.dim_c = dim_c
        self.use_locz_c = use_locz_c
        if self.n_classes > 0:
            self.eye = self.xp.eye(self.n_classes).astype(self.xp.float32)
        self.power_poly_c = power_poly_c
        if self.power_poly_c is not None:
            if channels_c is None:
                # # Initialize in the dimension of z.
                channels_c = [self.dim_c] * len(self.power_poly_c)
            elif isinstance(channels_c, int):
                channels_c = [channels_c] * len(self.power_poly_c)
            self.channels_c = channels_c
        self.inj_c_dim = self.channels_c[-1] if self.power_poly_c is not None else self.dim_c
        self.use_order_out_z = use_order_out_z
        self.order_bef_out_poly = order_bef_out_poly

        with self.init_scope():
            # # in this version, there is no categorical batch normalization.
            if type_norm == 'batch':
                bn1 = partial(L.BatchNormalization, use_gamma=True, use_beta=False)
            elif type_norm == 'instance':
                print('[Gen] Instance normalization in the generator.')
                bn1 = InstanceNormalization
            if self.typen_poly != self.type_norm and self.typen_poly == 'instance':
                print('[Gen] Using instance normalization for the FC layers *ONLY*.')
                bn2 = InstanceNormalization
            else:
                bn2 = bn1
            # # inpc_pol: the input size to the current polynomial; initialize on dimz.
            inpc_pol = dim_z
            # # iterate over all the polynomials (in the fully-connected part).
            print('[Gen] Number of product polynomials: {}.'.format(len(self.power_poly)))
            for id_poly in range(len(self.power_poly)):
                chp = self.channels[id_poly]
                m1 = '[Gen] Channels of the polynomial {}: {}, depth: {}.'
                print(m1.format(id_poly, chp, self.power_poly[id_poly]))
                # ensure that the current input channels match the expected.
                setattr(self, 'has_rsz{}'.format(id_poly), inpc_pol != chp)
                if inpc_pol != chp:
                    setattr(self, 'resize{}'.format(id_poly), Linear(inpc_pol, chp, nobias=not use_bias, initialW=w))
                # # now build the current polynomial (id_poly).
                for l in range(1, self.power_poly[id_poly] + self.add_one):
                    if l == 1 and id_poly > 0 and self.share_first:
                        # # in this case the layer will be shared with the first polynomial's
                        # # the first layer.
                        continue
                    setattr(self, 'l{}_{}'.format(id_poly, l), Linear(chp, chp, nobias=not use_bias, initialW=w))
                    if self.derivfc != 1 or l == 1:
                        # # define the layer for the second input (that is only if derivfc == 0 or we are at layer 1).
                        setattr(self, 'sec_inp{}_{}'.format(id_poly, l), L.Linear(self.inj_c_dim, chp, nobias=not use_bias, initialW=w))
                # # define the activation for this polynomial.
                actp = F.relu if self.activ_prod[id_poly] else lambda x: x
                setattr(self, 'activ{}'.format(id_poly), actp)
                if self.norm_after_poly:
                    # # define different batch normalizations for each polynomial repeated.
                    for repeat in range(self.repeat_poly[id_poly]):
                        setattr(self, 'bnp{}_r{}'.format(id_poly, repeat), bn2(chp))
                # # update input for the next polynomial.
                inpc_pol = int(chp)

            if bottom_width > 1:
                # # make a linear layer to transform to this shape.
                setattr(self, 'lin0', Linear(inpc_pol, inpc_pol * bottom_width ** 2, initialW=w))

            # # iterate over all layers (till the last) and save in self.
            for l in range(1, n_l + 1):
                # # define the input and the output names.
                ni, no = layer_d[l - 1], layer_d[l]
                Conv_sel = Conv0 if strides[l - 1] == 1 and ksizes[l - 1] == 3 and allow_conv else Conv
                conv_i = partial(Conv_sel, initialW=w, ksize=ksizes[l - 1], 
                                 stride=strides[l - 1], pad=paddings[l - 1])
                # # save the self.layer.
                setattr(self, 'l{}'.format(l), conv_i(ni, no))
                if self.skip_rep:
                    # # define some 1x1 convolutions iff the channels are not the same in
                    # # the input and the output. Otherwise, use the identity mapping.
                    func = (lambda x: x) if ni == no else Conv(ni, no, ksize=1)
                    setattr(self, 'skipch{}'.format(l), func)

            if self.order_out_poly is not None:
                self._init_conv_poly_out(Conv0, layer_d[n_l], w, inpc_pol, self.order_out_poly, 
                                         strat='oro', ch_in_zII=self.inj_c_dim)

            if self.order_bef_out_poly is not None:
                self._init_conv_poly_out(Conv0, layer_d[n_l], w, inpc_pol, self.order_bef_out_poly, 
                                         strat='orbo', ch_in_zII=self.inj_c_dim)

            # # save the last layer.
            # # In Deconv, we need to define a ksize (otherwise the dimensions unchanged). This 
            # # last layer leaves the spatial dimensions untouched.
            setattr(self, 'l{}'.format(n_l + 1), Conv(layer_d[n_l], out_ch, ksize=3, pad=1, initialW=w))
            self.n_channels.append(out_ch)
            if use_bn:
                # # set the batch norm (applied before the first layer conv).
                setattr(self, 'bn{}'.format(1), bn1(layer_d[0]))
                for l in range(2, self.n_l + 1):
                    # # set the batch norm for the layer.
                    setattr(self, 'bn{}'.format(l), bn1(layer_d[l - 1]))
            if self.mult_lat and use_localz:
                # # define the 'local' transformations of z (z local Linear).
                for l in range(1, n_l + 1):
                    ni, no = layer_d[0], layer_d[l]
                    setattr(self, 'locz{}'.format(l), Linear(ni, no))
                    if use_locz_c:
                        # # define the layer for the second input.
                        setattr(self, 'sec_inp_loc{}'.format(l), Linear(self.inj_c_dim, no, initialW=w))
            # # the condition to add a residual skip to the representation.
            self.skip1 = self.add_h_injection or self.add_h_poly or self.skip_rep

            if self.power_poly_c is not None:
                # # in this case, create a product of polynomials for the condition c;
                # # single variable polynomial to obtain higher-order correlations on c.
                self._create_poly_c(w, use_bias, self.dim_c)

    def _init_conv_poly_out(self, Conv, ch_out, init_w, ch_in_z, order_poly, strat='oro', ch_in_zII=None):
        """ Defines the polynomials connected to the output. """
        if ch_in_zII is None:
            ch_in_zII = ch_in_z
        for i in range(1, order_poly):
            setattr(self, '{}{}'.format(strat, i + 1), Conv(ch_out, ch_out, ksize=3, stride=1, pad=1, initialW=init_w))
            if self.use_order_out_z:
                setattr(self, '{}_zI{}'.format(strat, i + 1), L.Linear(ch_in_z, ch_out, initialW=init_w))
                setattr(self, '{}_zII{}'.format(strat, i + 1), L.Linear(ch_in_zII, ch_out, initialW=init_w))
        # assert not self.use_order_out_z, 'Not implemented yet (should be same dims as h in the output).'

    def _create_poly_c(self, initializer, use_bias, inpc_pol):
        """
        Define a polynomial on c variable (i.e. the discrete second variable).
        :param initializer: Chainer function for initialization.
        :param use_bias: Bool, whether to use bias.
        :param inpc_pol: int, the input channels to the polynomial c.
        """
        print('[Gen] Number of product polynomials (on c): {}.'.format(len(self.power_poly_c)))
        for id_poly in range(len(self.power_poly_c)):
            chp = self.channels_c[id_poly]
            m1 = '[Gen] Channels of the polynomial {}: {}, depth: {}.'
            print(m1.format(id_poly, chp, self.power_poly_c[id_poly]))
            # ensure that the current input channels match the expected.
            setattr(self, 'c_has_rsz{}'.format(id_poly), inpc_pol != chp)
            if inpc_pol != chp:
                setattr(self, 'c_resize{}'.format(id_poly), L.Linear(inpc_pol, chp, nobias=not use_bias,
                                                                     initialW=initializer))
            # # now build the current polynomial (id_poly).
            for l in range(1, self.power_poly_c[id_poly] + 1):
                setattr(self, 'c_l{}_{}'.format(id_poly, l), L.Linear(chp, chp, nobias=not use_bias,
                                                                      initialW=initializer))
            # # update input for the next polynomial.
            inpc_pol = int(chp)

    def _apply_poly_c(self, c):
        if self.power_poly_c is None:
            return c
        # # input_poly: the input variable to each polynomial; for the
        # # first, simply c, i.e. the conditional vector.
        input_poly = c + 0
        # # iterate over all the polynomials (length of channels many).
        for id_poly in range(len(self.channels_c)):
            # # ensure that the channels from previous polynomial are of the
            # # appropriate size.
            if getattr(self, 'c_has_rsz{}'.format(id_poly)):
                input_poly = getattr(self, 'c_resize{}'.format(id_poly))(input_poly)

            # # id_first: The index of the first FC layer; if we share it, it should
            # # be that of l0_1; otherwise l[id_poly]_1.
            id_first = id_poly
            h = getattr(self, 'c_l{}_1'.format(id_first))(input_poly)

            # # loop over the current polynomial layers and compute the
            # # output (for this polynomial).
            for layer in range(2, self.power_poly_c[id_poly] + 1):
                # # step 1: perform the hadamard product.
                if self.derivfc == 0:
                    z1 = getattr(self, 'c_l{}_{}'.format(id_poly, layer))(input_poly)
                    h = self.additional_noise(h)
                    if self.skip1:
                        h += z1 * h
                    else:
                        h = z1 * h
                elif self.derivfc == 1:
                    # # In this case, we assume that both A_i, B_i of
                    # # the original polygan derivation are identity matrices.
                    h1 = getattr(self, 'c_l{}_{}'.format(id_poly, layer))(h)
                    h1 = self.additional_noise(h1)
                    if self.skip1:
                        h += h1 * input_poly
                    else:
                        h = h1 * input_poly
            # # update the input for the next polynomial.
            input_poly = h + 0

        # # use the output of the products above as z.
        c = input_poly if not self.use_act_zgl else self.activation(input_poly)
        return c

    def return_injected(self, h, z, n_layer, mult_until_exec=None, y=None):
        """ Performs the local transformation of z and the injection. """
        # # check whether to skip the hadamard.
        skip_injection = False 
        if self.thresh_skip is not None and self.thresh_skip[n_layer-1] > 0:
            # # skip the hadamard, iff the random number is smaller than the threshold.
            skip_injection = np.random.uniform() < self.thresh_skip[n_layer-1]
        if not skip_injection and mult_until_exec is not None:
            skip_injection = mult_until_exec <= n_layer
        if self.mult_lat and not skip_injection:
            if self.use_localz:
                # # apply local transformation.
                z1 = getattr(self, 'locz{}'.format(n_layer))(z)
            else:
                z1 = z
            if self.use_locz_c:
                # # we need to include the product with the second input.
                c1 = getattr(self, 'sec_inp_loc{}'.format(n_layer))(y)
                z1 = z1 + c1
            # # appropriately reshape z for the elementwise multiplication.
            sh = h.shape
            z1 = F.reshape(z1, (sh[0], sh[1], 1))
            if self.normalize_preinject:
                z1 /= F.sqrt(F.mean(z1 * z1, axis=1, keepdims=True) + 1e-8)
            z2 = F.repeat(z1, sh[3] * sh[2], axis=2)
            z2 = F.reshape(z2, sh)
            ret = h * z2 + h if self.add_h_injection else h * z2
            return ret
        return h

    def prod_poly_FC(self, z, mult_until_exec, batchsize=None, y=None):
        """
        Performs the products of polynomials for the fully-connected part.
        """
        # # input_poly: the input variable to each polynomial; for the 
        # # first, simply z, i.e. the noise vector.
        input_poly = z + 0
        # # iterate over all the polynomials (length of channels many).
        for id_poly in range(len(self.channels)):
            # # ensure that the channels from previous polynomial are of the
            # # appropriate size.
            if getattr(self, 'has_rsz{}'.format(id_poly)):
                input_poly = getattr(self, 'resize{}'.format(id_poly))(input_poly)

            # # id_first: The index of the first FC layer; if we share it, it should
            # # be that of l0_1; otherwise l[id_poly]_1.
            id_first = id_poly if not self.share_first else 0
            # # order of the current polynomial.
            order = self.power_poly[id_poly] + self.add_one
            # # condition to use the second recursive term.
            sec_rec = order > 2 and self.skip1 and self.use_sec_recurs
            # # Repeat each polynomial repeat_poly times (if repeat==1, then only running once).
            for repeat in range(self.repeat_poly[id_poly]):
                c1 = getattr(self, 'sec_inp{}_1'.format(id_poly))(y)
                h = getattr(self, 'l{}_1'.format(id_first))(input_poly) + c1
                if sec_rec:
                    all_reps = []

                # # loop over the current polynomial layers and compute the 
                # # output (for this polynomial). 
                for layer in range(2, order):
                    if layer <= mult_until_exec:
                        # # step 1: perform the hadamard product.
                        if self.derivfc == 0:
                            z1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(input_poly)
                            c1 = getattr(self, 'sec_inp{}_{}'.format(id_poly, layer))(y)
                            h = self.additional_noise(h)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += (z1 + c1) * h
                            else:
                                h = (z1 + c1) * h
                        elif self.derivfc == 1:
                            # # In this case, we assume that both A_i, B_i of
                            # # the original polygan derivation are identity matrices.
                            h1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(h)
                            h1 = self.additional_noise(h1)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += h1 * (input_poly + c1)
                            else:
                                h = h1 * (input_poly + c1)
                        if sec_rec and layer > 2:
                            h = h + self.sign_srec * all_reps[-2]
                    # # step 2: activation.
                    h = getattr(self, 'activ{}'.format(id_poly))(h)
                # # if we included the second recursive term, add back in the final representation.
                if sec_rec:
                    for rep in all_reps:
                        h = h - self.sign_srec * rep
                if self.norm_after_poly:
                    # # apply a different BN for every repeat time of the poly.
                    if y is None or self.typen_poly == 'instance':
                        h = getattr(self, 'bnp{}_r{}'.format(id_poly, repeat))(h)
                    else:
                        h = getattr(self, 'bnp{}_r{}'.format(id_poly, repeat))(h, y)
                # # update the input for the next polynomial.
                input_poly = h + 0

        # # use the output of the products above as z.
        z = input_poly if not self.use_act_zgl else self.activation(input_poly)
        return z

    def additional_noise(self, rep):
        """ Sample additional (additive) noise and add it to the representation. """
        sh = rep.shape
        if self.use_add_noise:
            noise = sample_continuous(1, sh[0], distribution=self.distribution, xp=self.xp)
            n1 = F.repeat(noise, int(np.prod(sh[1:])), axis=1)
            n2 = F.reshape(n1, sh)
            return rep + n2
        return rep

    def run_poly_out(self, h, order_poly, z=None, strat='oro', zII=None):
        """ Run the polynomials connected to the output. """
        z0 = h + 0
        for i in range(1, order_poly):
            h1 = getattr(self, '{}{}'.format(strat, i + 1))(h)
            h1 = self.additional_noise(h1)
            if self.use_order_out_z:
                z1 = getattr(self, '{}_zI{}'.format(strat, i + 1))(z)
                z2 = getattr(self, '{}_zII{}'.format(strat, i + 1))(zII)

                z4 = z1 + z2
                sh = h1.shape
                z4 = F.reshape(z4, (sh[0], sh[1], 1))
                z5 = F.repeat(z4, sh[3] * sh[2], axis=2)
                z5 = F.reshape(z5, sh)
                if self.skip_rep:
                    # # model3 polynomial.
                    h += (z0 + z5) * h1
                else:
                    h = (z0 + z5) * h1
            else:
                if self.skip_rep:
                    # # model3 polynomial.
                    h += z0 * h1
                else:
                    h = z0 * h1
        return h

    def __call__(self, batchsize=None, y=None, z=None, mult_until_exec=None, y2=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None and self.n_classes > 0:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform", xp=self.xp)
        elif y2 is not None:
            y = y2
        if self.n_classes > 0:
            # # convert the class to one hot.
            cl = chainer.cuda.to_cpu(y)
            y = self.xp.array(self.eye[cl])
        activ = self.activation

        # # create a polynomial on y (condition).
        y = self._apply_poly_c(y)
        # # mult_until_exec: If set, we perform the multiplications until that layer.
        # # In the product of polynomials, it applies the same rule for *every*
        # # polynomial. E.g. if mult_until_exec == 2 it will perform the hadamard
        # # products until second order terms in every polynomial.
        if mult_until_exec is None:
            mult_until_exec = 10000
        z = self.prod_poly_FC(z, mult_until_exec, batchsize=batchsize, y=y)

        h = z + 0
        if self.bottom_width > 1:
            h = getattr(self, 'lin0')(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))

        # # loop over the layers and get the layers along with the
        # # normalizations per layer.
        for l in range(1, self.n_l + 1):
            if self.skip_rep:
                h_hold = h + 0
            if self.use_bn:
                h = getattr(self, 'bn{}'.format(l))(h)
            h = activ(getattr(self, 'l{}'.format(l))(h))
            h = self.additional_noise(h)
            h = self.return_injected(h, z, l, mult_until_exec=mult_until_exec, y=y)
            if self.skip_rep:
                # # transform the channels of h_hold if required.
                h_hold = getattr(self, 'skipch{}'.format(l))(h_hold)
                # # upsample if required.
                if h_hold.shape[-1] != h.shape[-1]:
                    h_hold = _upsample(h_hold)
                h += h_hold
        if self.order_bef_out_poly is not None:
            h = self.run_poly_out(h, self.order_bef_out_poly, z=z, strat='orbo', zII=y)
        if self.order_out_poly is not None:
            h = self.run_poly_out(h, self.order_out_poly, z=z, strat='oro', zII=y)

        # # last layer (no activation).
        output = getattr(self, 'l{}'.format(self.n_l + 1))(h)
        out = self.out_act(output)
        return out

    def __str__(self, **kwargs):
        m1 = 'Layers: {}.\t Info for channels: {}.'
        str1 = m1.format(self.n_l, self.n_channels)
        return str1


class ProdPoly2InVarConvGeneratorCont(chainer.Chain):
    def __init__(self, layer_d, use_bn=False, sn=False, out_ch=2, n_classes=0, bottom_width=4,
                 distribution='uniform', ksizes=None, strides=None, paddings=None,
                 use_out_act=False, channels=None, power_poly=[4, 3],
                 derivfc=0, activ_prod=0, use_bias=True, dim_z=128, train=True, activ=None,
                 add_h_injection=False, add_h_poly=False, norm_after_poly=False,
                 normalize_preinject=False, use_act_zgl=False, allow_conv=False,
                 repeat_poly=1, share_first=False, type_norm='batch', typen_poly='batch',
                 skip_rep=False, order_out_poly=None, use_activ=False, thresh_skip=None,
                 use_sec_recurs=False, sign_srec=-1, use_add_noise=False, glorot_scale=1.0,
                 skip_rep_c=False, order_in_c=None, channels_in_c=3, use_order_out_z=False,
                 order_in_z=None, global_skip=False, order_bef_out_poly=None, downsample_c=None, 
                 use_order_out_zI=False):
        """
        Initializes the product polynomial generator with 2 inputs (both of them continuous). Specifically, 
        z_II assumed of image-type.
        :param layer_d: List of all layers' inputs/outputs channels (including
                        the input to the last output).
        :param use_bn: Bool, whether to use batch normalization.
        :param sn: Bool, if True use spectral normalization.
        :param out_ch: int, Output channels of the generator.
        :param ksizes: List of the kernel sizes per layer.
        :param strides: List of the stride per layer.
        :param paddings: List of the padding per layer (int).
        :param activ: str or None: The activation to use.
        :param bottom_width: int, spatial resolution to reshape the init noise.
        :param use_out_act: bool, if True, it uses a tanh as output activation.
        :param repeat_poly: int or list: List of how many times to repeat each
               polynomial (original with FC layers, the last is NOT repeated).
        :param share_first: bool. If True, it shares the first FC layer in the
               FC polynomials. It assumes the respective FC layers are of the same size.
        :param type_norm: str. The type of normalization to be used, i.e. 'batch' for
               BN, 'instance' for instance norm.
        :param typen_poly: str. The type of normalization to be used ONLY
               for the FC layers.
        :param skip_rep: bool. If True, it uses a skip connection before the layer to
               the next, i.e. in practice it directly skips the convolution/fc plus hadamard
               product of this layer. See derivation 3.
        :param order_out_poly: int or None. If int, a polynomial of the order is used
               in the output, i.e. before the final convolution.
        :param use_activ: bool. If True, use activations in the main deconvolutional part.
        :param thresh_skip: None, int or list. It represents the probability of skipping the
               hadamard product in the deconvolution part (i.e. main polynomial). This is per
               layer; e.g. if 0.3, this means that approx. 30% of the time this hadamard will
               be skipped. It should have values in the [0, 1]. In inference, it should be
               deactivated.
        :param use_sec_recurs: bool, default False. If True, it uses the second recursive term,
               i.e. x_(n+1)=f(x_n, x_(n-1)). This is used only in the derivation 3 setting. Also,
               for practical reasons, use only for the FC layers when their order > 3.
        :param use_add_noise: bool, default False. If True, it draws additive Gaussian noise (scalar)
               after each convolutional layer.
        :param skip_rep_c: bool, default False. If True, use skip in the c related polynomials (model 3).
        :param order_in_c: int or None. If not None, use a polynomial of that order in c (i.e. before the
               multivariate part). 
        :param channels_in_c: int. The input channels of the continuous input c. That is c is assumed to be
               an image already, so provide the channels, e.g. 3 for RGB.
        :param use_order_out_z: bool. If True, convert the order_out_poly into a 2-variable  polynomial.
        :param order_in_z: int or None. If int, add a polynomial for upsampling the noise z to match
               the dimensions of input c. That is, if the bottom_width is smaller than c spatial resolution,
               this order_in_z upsamples the z noise.
        :param order_bef_out_poly: int or None. If int, a polynomial of the order is used
               before the output polynomial, i.e. see order_out_poly above.
        """
        super(ProdPoly2InVarConvGeneratorCont, self).__init__()
        self.n_l = n_l = len(layer_d) - 1
        w = chainer.initializers.GlorotUniform(glorot_scale)
        if sn:
            Conv = SNConvolution2D
            Linear = SNLinear
            raise RuntimeError('Not implemented!')
        else:
            Conv = L.Deconvolution2D
            Linear = L.Linear
        Conv0 = L.Convolution2D
        # # initialize args not provided.
        if ksizes is None:
            ksizes = [4] * n_l
        if strides is None:
            strides = [2] * n_l
        if paddings is None:
            paddings = [1] * n_l
        # # the length of the ksizes and the strides is only for the conv layers, hence
        # # it should be one number short of the layer_d.
        assert len(ksizes) == len(strides) == len(paddings) == n_l
        # # save in self, several useful properties.
        self.use_bn = use_bn
        self.n_channels = layer_d
        self.distribution = distribution
        # # Define dim_z.
        assert isinstance(dim_z, int)
        self.dim_z = dim_z
        self.train = train
        # # bottom_width: The default starting spatial resolution of the convolutions.
        self.bottom_width = bottom_width
        self.n_classes = n_classes
        if activ is not None:
            activ = getattr(F, activ)
        elif use_activ:
            activ = F.relu
        else:
            activ = lambda x: x
        self.activ = self.activation = activ
        self.out_act = F.tanh if use_out_act else lambda x: x
        # # Set the add_one to True out of consistency with the other
        # # implementation (resnet-based).
        self.add_one = 1
        self.add_h_injection = add_h_injection
        self.normalize_preinject = normalize_preinject
        self.use_bias = use_bias
        self.add_h_poly = add_h_poly
        self.power_poly = power_poly
        if channels is None:
            # # Initialize in the dimension of z.
            channels = [dim_z] * len(self.power_poly)
        self.channels = channels
        if isinstance(activ_prod, int):
            activ_prod = [activ_prod] * len(self.power_poly)
        self.activ_prod = activ_prod
        if isinstance(repeat_poly, int):
            repeat_poly = [repeat_poly] * len(self.power_poly)
        self.repeat_poly = repeat_poly
        assert len(self.power_poly) == len(self.activ_prod) == len(self.channels)
        assert len(self.power_poly) == len(self.repeat_poly)
        # # If True, it compensates for python open interval in the end of range.
        self.derivfc = derivfc
        self.norm_after_poly = norm_after_poly
        # # whether to use an activation in the global transformation.
        self.use_act_zgl = use_act_zgl and use_activ
        self.share_first = share_first
        self.type_norm = return_norm(type_norm)
        self.typen_poly = return_norm(typen_poly)
        self.skip_rep = skip_rep
        # # optionally adding one 'output polynomial', before making the final output convolution.
        self.order_out_poly = order_out_poly
        # # set the threshold for skipping the injection in the deconvolution part.
        if isinstance(thresh_skip, float) and self.train:
            self.thresh_skip = [0] + [thresh_skip] * (n_l - 1)
            print('[Gen] Skip probabilities: {}.'.format(self.thresh_skip))
        else:
            self.thresh_skip = thresh_skip
        self.use_sec_recurs = use_sec_recurs and skip_rep
        self.sign_srec = sign_srec
        self.use_add_noise = use_add_noise
        # # add for the 2var continuous case.
        self.skip_rep_c = skip_rep_c
        self.order_in_c = order_in_c
        self.channels_in_c = channels_in_c
        assert self.bottom_width > 1, 'in this implementation we require that to be the case.'
        # # use_order_out_z: if True, convert the order_out_poly into a 2-variable  polynomial.
        self.use_order_out_z = use_order_out_z
        # # If True, the output polynomial(s) will be also correlated with zI (i.e. noise).
        self.use_order_out_zI = use_order_out_zI
        self.order_in_z = order_in_z
        self.global_skip = global_skip
        self.order_bef_out_poly = order_bef_out_poly
        if downsample_c is None:
            downsample_c = [-1] * (self.n_l + 1)
        self.downsample_c = downsample_c

        with self.init_scope():
            if type_norm == 'batch':
                bn1 = partial(L.BatchNormalization, use_gamma=True, use_beta=False)
            elif type_norm == 'instance':
                print('[Gen] Instance normalization in the generator.')
                bn1 = InstanceNormalization
            if self.typen_poly != self.type_norm and self.typen_poly == 'instance':
                print('[Gen] Using instance normalization for the FC layers *ONLY*.')
                bn2 = InstanceNormalization
            else:
                bn2 = bn1
            # # inpc_pol: the input size to the current polynomial; initialize on dimz.
            inpc_pol = dim_z
            # # iterate over all the polynomials (in the fully-connected part).
            print('[Gen] Number of product polynomials: {}.'.format(len(self.power_poly)))
            for id_poly in range(len(self.power_poly)):
                chp = self.channels[id_poly]
                m1 = '[Gen] Channels of the polynomial {}: {}, depth: {}.'
                print(m1.format(id_poly, chp, self.power_poly[id_poly]))
                # ensure that the current input channels match the expected.
                setattr(self, 'has_rsz{}'.format(id_poly), inpc_pol != chp)
                if inpc_pol != chp:
                    setattr(self, 'resize{}'.format(id_poly), Linear(inpc_pol, chp, nobias=not use_bias, initialW=w))
                # # now build the current polynomial (id_poly).
                for l in range(1, self.power_poly[id_poly] + self.add_one):
                    if l == 1 and id_poly > 0 and self.share_first:
                        # # in this case the layer will be shared with the first polynomial's
                        # # the first layer.
                        continue
                    setattr(self, 'l{}_{}'.format(id_poly, l), Linear(chp, chp, nobias=not use_bias, initialW=w))
                # # define the activation for this polynomial.
                actp = F.relu if self.activ_prod[id_poly] else lambda x: x
                setattr(self, 'activ{}'.format(id_poly), actp)
                if self.norm_after_poly:
                    # # define different batch normalizations for each polynomial repeated.
                    for repeat in range(self.repeat_poly[id_poly]):
                        setattr(self, 'bnp{}_r{}'.format(id_poly, repeat), bn2(chp))
                # # update input for the next polynomial.
                inpc_pol = int(chp)

            if bottom_width > 1:
                # # make a linear layer to transform to this shape.
                setattr(self, 'lin0', Linear(inpc_pol, inpc_pol * bottom_width ** 2, initialW=w))

            # # in this version, we use a hardcoded upsample.
            self.upsamples = [0]
            # # iterate over all layers (till the last) and save in self.
            for l in range(1, n_l + 1):
                # # define the input and the output names.
                ni, no = layer_d[l - 1], layer_d[l]
                Conv_sel = Conv0 if strides[l - 1] == 1 and ksizes[l - 1] == 3 and allow_conv else Conv
                conv_i = partial(Conv_sel, initialW=w, ksize=ksizes[l - 1],
                                 stride=strides[l - 1], pad=paddings[l - 1])
                self.upsamples.append(ksizes[l - 1] > 3)
                # # save the self.layer.
                setattr(self, 'l{}'.format(l), conv_i(ni, no))
                # # in this version the locz and the convolution for the continuous var are here.
                setattr(self, 'locz{}'.format(l), Conv0(layer_d[0], no, ksize=3, stride=1, pad=1, initialW=w))
                setattr(self, 'locy{}'.format(l), Conv0(layer_d[0], no, ksize=3, stride=1, pad=1, initialW=w))
                if self.skip_rep:
                    # # define some 1x1 convolutions iff the channels are not the same in
                    # # the input and the output. Otherwise, use the identity mapping.
                    func = (lambda x: x) if ni == no else Conv(ni, no, ksize=1)
                    setattr(self, 'skipch{}'.format(l), func)

            if self.order_out_poly is not None:
                self._init_conv_poly_out(Conv0, layer_d[n_l], w, inpc_pol, self.order_out_poly, strat='oro')

            if self.order_bef_out_poly is not None:
                self._init_conv_poly_out(Conv0, layer_d[n_l], w, inpc_pol, self.order_bef_out_poly, strat='orbo')

            # # in case the noise input and the continuous input c are not same dimensions, add
            # # upsampling layers for the noise input.
            if self.order_in_z is not None and self.order_in_z > 0:
                ch1 = inpc_pol
                for i in range(1, self.order_in_z + 1):
                    setattr(self, 'ori_z{}'.format(i), Conv0(ch1, ch1, ksize=3, stride=1, pad=1, initialW=w))

            if self.order_in_c is not None and self.order_in_c > 0:
                ch_in, ch1 = channels_in_c, inpc_pol
                setattr(self, 'ori_c{}'.format(1), Conv0(ch_in, ch1, ksize=3, stride=1, pad=1, initialW=w))
                for i in range(2, self.order_in_c + 1):
                    setattr(self, 'ori_c{}'.format(i), Conv0(ch1, ch1, ksize=3, stride=1, pad=1, initialW=w))

            # # save the last layer.
            # # In Deconv, we need to define a ksize (otherwise the dimensions unchanged). This
            # # last layer leaves the spatial dimensions untouched.
            setattr(self, 'l{}'.format(n_l + 1), Conv(layer_d[n_l], out_ch, ksize=3,
                                                      pad=1, initialW=w))
            self.n_channels.append(out_ch)
            if use_bn:
                # # set the batch norm (applied before the first layer conv).
                setattr(self, 'bn{}'.format(1), bn1(layer_d[0]))
                for l in range(2, self.n_l + 1):
                    # # set the batch norm for the layer.
                    setattr(self, 'bn{}'.format(l), bn1(layer_d[l - 1]))
            # # the condition to add a residual skip to the representation.
            self.skip1 = self.add_h_injection or self.add_h_poly or self.skip_rep

    def _init_conv_poly_out(self, Conv, ch_out, init_w, ch_in_z, order_poly, strat='oro'):
        """ Defines the polynomials connected to the output. """
        print('[Gen] strat={}, order={}'.format(strat, order_poly))
        for i in range(1, order_poly):
            setattr(self, '{}{}'.format(strat, i + 1), Conv(ch_out, ch_out, ksize=3, stride=1, pad=1, initialW=init_w))
            if self.use_order_out_z:
                setattr(self, '{}_z{}'.format(strat, i + 1), Conv(ch_in_z, ch_out, ksize=3, stride=1,
                                                                  pad=1, initialW=init_w))
            if self.use_order_out_zI:
                setattr(self, '{}_zI{}'.format(strat, i + 1), L.Linear(ch_in_z, ch_out, initialW=init_w))

    def prod_poly_FC(self, z, mult_until_exec, batchsize=None, y=None):
        """
        Performs the products of polynomials for the fully-connected part.
        """
        # # input_poly: the input variable to each polynomial; for the
        # # first, simply z, i.e. the noise vector.
        input_poly = z + 0
        # # iterate over all the polynomials (length of channels many).
        for id_poly in range(len(self.channels)):
            # # ensure that the channels from previous polynomial are of the
            # # appropriate size.
            if getattr(self, 'has_rsz{}'.format(id_poly)):
                input_poly = getattr(self, 'resize{}'.format(id_poly))(input_poly)

            # # id_first: The index of the first FC layer; if we share it, it should
            # # be that of l0_1; otherwise l[id_poly]_1.
            id_first = id_poly if not self.share_first else 0
            # # order of the current polynomial.
            order = self.power_poly[id_poly] + self.add_one
            # # condition to use the second recursive term.
            sec_rec = order > 2 and self.skip1 and self.use_sec_recurs
            # # Repeat each polynomial repeat_poly times (if repeat==1, then only running once).
            for repeat in range(self.repeat_poly[id_poly]):
                h = getattr(self, 'l{}_1'.format(id_first))(input_poly)
                if sec_rec:
                    all_reps = []

                # # loop over the current polynomial layers and compute the
                # # output (for this polynomial).
                for layer in range(2, order):
                    if layer <= mult_until_exec:
                        # # step 1: perform the hadamard product.
                        if self.derivfc == 0:
                            z1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(input_poly)
                            h = self.additional_noise(h)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += z1 * h
                            else:
                                h = z1 * h
                        elif self.derivfc == 1:
                            # # In this case, we assume that both A_i, B_i of
                            # # the original polygan derivation are identity matrices.
                            h1 = getattr(self, 'l{}_{}'.format(id_poly, layer))(h)
                            h1 = self.additional_noise(h1)
                            if self.skip1:
                                if sec_rec:
                                    all_reps.append(h + 0)
                                h += h1 * input_poly
                            else:
                                h = h1 * input_poly
                        if sec_rec and layer > 2:
                            h = h + self.sign_srec * all_reps[-2]
                    # # step 2: activation.
                    h = getattr(self, 'activ{}'.format(id_poly))(h)
                # # if we included the second recursive term, add back in the final representation.
                if sec_rec:
                    for rep in all_reps:
                        h = h - self.sign_srec * rep
                if self.norm_after_poly:
                    # # apply a different BN for every repeat time of the poly.
                    h = getattr(self, 'bnp{}_r{}'.format(id_poly, repeat))(h)
                # # update the input for the next polynomial.
                input_poly = h + 0

        # # use the output of the products above as z.
        z = input_poly if not self.use_act_zgl else self.activation(input_poly)
        return z

    def additional_noise(self, rep):
        """ Sample additional (additive) noise and add it to the representation. """
        sh = rep.shape
        if self.use_add_noise:
            noise = sample_continuous(1, sh[0], distribution=self.distribution, xp=self.xp)
            n1 = F.repeat(noise, int(np.prod(sh[1:])), axis=1)
            n2 = F.reshape(n1, sh)
            return rep + n2
        return rep

    def run_poly_out(self, h, order_poly, z=None, strat='oro', zI=None):
        """ Run the polynomials connected to the output. """
        z0 = h + 0
        for i in range(1, order_poly):
            h1 = getattr(self, '{}{}'.format(strat, i + 1))(h)
            h1 = self.additional_noise(h1)
            if self.use_order_out_zI:
                z2 = getattr(self, '{}_zI{}'.format(strat, i + 1))(zI)
                sh = h1.shape
                z2 = F.reshape(z2, (sh[0], sh[1], 1))
                z3 = F.repeat(z2, sh[3] * sh[2], axis=2)
                z3 = F.reshape(z3, sh)
            if self.use_order_out_z:
                z1 = getattr(self, '{}_z{}'.format(strat, i + 1))(z)
                if self.use_order_out_zI:
                    z1 += z3
                if self.skip_rep:
                    # # model3 polynomial.
                    h += (z0 + z1) * h1
                else:
                    h = (z0 + z1) * h1
            elif self.use_order_out_zI:
                if self.skip_rep:
                    # # model3 polynomial.
                    h += (z0 + z3) * h1
                else:
                    h = (z0 + z3) * h1
            else:
                if self.skip_rep:
                    # # model3 polynomial.
                    h += z0 * h1
                else:
                    h = z0 * h1
        return h

    def __call__(self, batchsize=None, y=None, y2=None, z=None, mult_until_exec=None, **kwargs):
        if z is None:
            z = sample_continuous(self.dim_z, batchsize, distribution=self.distribution, xp=self.xp)
        if y is None and self.n_classes > 0:
            y = sample_categorical(self.n_classes, batchsize, distribution="uniform",
                                   xp=self.xp) if self.n_classes > 0 else None
        activ = self.activation
        if self.global_skip:
            y2_keep = y2 + 0

        # # mult_until_exec: If set, we perform the multiplications until that layer.
        # # In the product of polynomials, it applies the same rule for *every*
        # # polynomial. E.g. if mult_until_exec == 2 it will perform the hadamard
        # # products until second order terms in every polynomial.
        if mult_until_exec is None:
            mult_until_exec = 10000
        z_org = self.prod_poly_FC(z, mult_until_exec, batchsize=batchsize, y=y)
        z = z_org + 0
        h = z_org + 0
        if self.bottom_width > 1:
            h = getattr(self, 'lin0')(h)
        h = F.reshape(h, (h.shape[0], -1, self.bottom_width, self.bottom_width))

        if self.order_in_z is not None and self.order_in_z > 0:
            z0 = self.ori_z1(h)
            z0 = _upsample(z0)
            hz = z0 + 0
            for i in range(2, self.order_in_z + 1):
                h1 = getattr(self, 'ori_z{}'.format(i))(hz)
                if self.skip_rep_c:
                    # # model3 polynomial.
                    hz += z0 * h1
                else:
                    hz = z0 * h1
                # # the goal of this polynomial is to upsample z, so do that.
                z0 = _upsample(z0)
                hz = _upsample(hz)
            h = hz + 0

        if self.order_in_c is not None and self.order_in_c > 0:
            z0 = self.ori_c1(y2)
            hy = z0 + 0
            for i in range(2, self.order_in_c + 1):
                h1 = getattr(self, 'ori_c{}'.format(i))(hy)
                if self.skip_rep_c:
                    # # model3 polynomial.
                    hy += z0 * h1
                else:
                    hy = z0 * h1
            y2 = hy + 0
        # assert np.all(np.array(y2.shape) == np.array(h.shape)) or np.max(self.downsample_c) > 0

        z = h + 0
        # # loop over the layers and get the layers along with the
        # # normalizations per layer.
        for l in range(1, self.n_l + 1):
            if self.skip_rep:
                h_hold = h + 0
            if self.use_bn:
                h = getattr(self, 'bn{}'.format(l))(h)
            h = activ(getattr(self, 'l{}'.format(l))(h))
            h = self.additional_noise(h)
            # # required upsamplings first.
            if self.upsamples[l]:
                z = _upsample(z)
                if self.downsample_c[l] < 0:
                    # # upsample y2, only in the case that the respective downsampling is zero.
                    y2 = _upsample(y2)
            z1 = activ(getattr(self, 'locz{}'.format(l))(z))
            if self.downsample_c[l] > 0:
                # # downsample y2 before convolving.
                y2_1 = _downsample(y2)
                for i in range(2, self.downsample_c[l] + 1):
                    y2_1 = _downsample(y2_1)
                y1 = activ(getattr(self, 'locy{}'.format(l))(y2_1))
            else:
                y1 = activ(getattr(self, 'locy{}'.format(l))(y2))
            h = (y1 + z1) * h
            if self.skip_rep:
                # # transform the channels of h_hold if required.
                h_hold = getattr(self, 'skipch{}'.format(l))(h_hold)
                # # upsample if required.
                if h_hold.shape[-1] != h.shape[-1]:
                    h_hold = _upsample(h_hold)
                h += h_hold
        if self.order_bef_out_poly is not None:
            # # z should be the same dimensionality as h; here achieved in the Polu_U (i.e. poly above), since
            # # the z is continuously upsampled there.
            h = self.run_poly_out(h, self.order_bef_out_poly, z=z, strat='orbo', zI=z_org)
        if self.order_out_poly is not None:
            h = self.run_poly_out(h, self.order_out_poly, z=z, strat='oro', zI=z_org)
        # # last layer (no activation).
        output = getattr(self, 'l{}'.format(self.n_l + 1))(h)
        if self.global_skip:
            # # essentially, it should learn the residual.
            output += y2_keep
        out = self.out_act(output)
        return out

