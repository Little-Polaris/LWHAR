from collections import namedtuple
from itertools import chain, starmap
from operator import itemgetter

import numpy
import torch
from gph import ripser_parallel
from torch import nn


def nesting_level(x):
    """Calculate nesting level of a list of objects.

    To convert between sparse and dense representations of topological
    features, we need to determine the nesting level of an input list.
    The nesting level is defined as the maximum number of times we can
    recurse into the object while still obtaining lists.

    Parameters
    ----------
    x : list
        Input list of objects.

    Returns
    -------
    int
        Nesting level of `x`. If `x` has no well-defined nesting level,
        for example because `x` is not a list of something, will return
        `0`.

    Notes
    -----
    This function is implemented recursively. It is therefore a bad idea
    to apply it to objects with an extremely high nesting level.

    Examples
    --------
    >>> nesting_level([1, 2, 3])
    1

    >>> nesting_level([[1, 2], [3, 4]])
    2
    """
    # This is really only supposed to work with lists. Anything fancier,
    # for example a `torch.tensor`, can already be used as a dense data
    # structure.
    if not isinstance(x, list):
        return 0

    # Empty lists have a nesting level of 1.
    if len(x) == 0:
        return 1
    else:
        return max(nesting_level(y) for y in x) + 1

def make_tensor_from_persistence_information(
    pers_info, extract_generators=False
):
    """Convert (sequence) of persistence information entries to tensor.

    This function converts instance(s) of :class:`PersistenceInformation`
    objects into a single tensor. No padding will be performed. A client
    may specify what type of information to extract from the object. For
    instance, by default, the function will extract persistence diagrams
    but this behaviour can be changed by setting `extract_generators` to
    `true`.

    Parameters
    ----------
    pers_info : :class:`PersistenceInformation` or iterable thereof
        Input persistence information object(s). The function is able to
        handle both single objects and sequences. This has no bearing on
        the length of the returned tensor.

    extract_generators : bool
        If set, extracts generators instead of persistence diagram from
        `pers_info`.

    Returns
    -------
    torch.tensor
        Tensor of shape `(n, 3)`, where `n` is the sum of all features,
        over all dimensions in the input `pers_info`. Each triple shall
        be of the form `(creation, destruction, dim)` for a persistence
        diagram. If the client requested generators to be returned, the
        first two entries of the triple refer to *indices* with respect
        to the input data set. Depending on the algorithm employed, the
        meaning of these indices can change. Please refer to the module
        used to calculate persistent homology for more details.
    """
    # Looks a little bit cumbersome, but since `namedtuple` is iterable
    # as well, we need to ensure that we are actually dealing with more
    # than one instance here.
    if len(pers_info) > 1 and not isinstance(
        pers_info[0], PersistenceInformation
    ):
        pers_info = [pers_info]

    # TODO: This might not always work since the size of generators
    # changes in different dimensions.
    if extract_generators:
        pairs = torch.cat(
            [torch.as_tensor(x.pairing, dtype=torch.float) for x in pers_info],
        ).long()
    else:
        pairs = torch.cat(
            [torch.as_tensor(x.diagram, dtype=torch.float) for x in pers_info],
        ).float()

    dimensions = torch.cat(
        [
            torch.as_tensor(
                [x.dimension] * len(x.diagram),
                dtype=torch.long,
                device=x.diagram.device,
            )
            for x in pers_info
        ]
    )

    result = torch.column_stack((pairs, dimensions))
    return result

def make_tensor(x):
    """Create dense tensor representation from sparse inputs.

    This function turns sparse inputs of :class:`PersistenceInformation`
    objects into 'dense' tensor representations, thus providing a useful
    integration into differentiable layers.

    The dimension of the resulting tensor depends on maximum number of
    topological features, summed over all dimensions in the data. This
    is similar to the format in `giotto-ph`.

    Parameters
    ----------
    x : list of (list of ...) :class:`PersistenceInformation`
        Input, consisting of a (potentially) nested list of
        :class:`PersistenceInformation` objects as obtained
        from a persistent homology calculation module, such
        as :class:`VietorisRipsComplex`.

    Returns
    -------
    torch.tensor
        Dense tensor representation of `x`. The output is best
        understood by considering some examples: given a batch
        obtained from :class:`VietorisRipsComplex`, our tensor
        will have shape `(B, N, 3)`. `B` is the batch size and
        `N` is the sum of maximum lengths of diagrams relative
        to this batch. Each entry will consist of a creator, a
        destroyer, and a dimension. Dummy entries, used to pad
        the batch, can be detected as `torch.nan`.
    """
    level = nesting_level(x)

    # Internal utility function for calculating the length of the output
    # tensor. This is required to ensure that all inputs can be *merged*
    # into a single output tensor.
    def _calculate_length(x, level):

        # Simple base case; should never occur in practice but let's be
        # consistent here.
        if len(x) == 0:
            return 0

        # Each `chain.from_iterable()` removes an additional layer of
        # nesting. We only have to start from level 2; we get level 1
        # for free because we can always iterate over a list.
        for i in range(2, level + 1):
            x = chain.from_iterable(x)

        # Collect information that we need to create the full tensor. An
        # entry of the resulting list contains the length of the diagram
        # and the dimension, making it possible to derive padding values
        # for all entries.
        M = list(map(lambda a: (len(a.diagram), a.dimension), x))

        # Get maximum dimension
        dim = max(M, key=itemgetter(1))[1]

        # Get *sum* of maximum number of entries for each dimension.
        # This is calculated over all batches.
        N = sum(
            [
                max([L for L in M if L[1] == d], key=itemgetter(0))[0]
                for d in range(dim + 1)
            ]
        )

        return N

    # Auxiliary function for padding tensors with `torch.nan` to
    # a specific dimension. Will always return a `list`; we turn
    # it into a tensor depending on the call level.
    def _pad_tensors(tensors, N, value=torch.nan):
        return list(
            map(
                lambda t: torch.nn.functional.pad(
                    t, (0, 0, N - len(t), 0), mode="constant", value=value
                ),
                tensors,
            )
        )

    N = _calculate_length(x, level)

    # List of lists: the first axis is treated as the batch axis, while
    # the second axis is treated as the dimension of diagrams or pairs.
    # This also handles ordinary lists, which will result in a batch of
    # size 1.
    if level <= 2:
        tensors = [
            make_tensor_from_persistence_information(pers_infos)
            for pers_infos in x
        ]

        # Pad all tensors to length N in the first dimension, then turn
        # them into a batch.
        result = torch.stack(_pad_tensors(tensors, N))
        return result

    # List of lists of lists: this indicates image-based data, where we
    # also have a set of tensors for each channel. The internal layout,
    # i.e. our input, has the following structure:
    #
    # B x C x D
    #
    # Each variable being the length of the respective list. We want an
    # output of the following shape:
    #
    # B x C x N x 3
    #
    # Here, `N` is the maximum length of an individual persistence
    # information object.
    else:
        tensors = [
            [
                make_tensor_from_persistence_information(pers_infos)
                for pers_infos in batch
            ]
            for batch in x
        ]

        # Pad all tensors to length N in the first dimension, then turn
        # them into a batch. We first stack over channels (inner), then
        # over the batch (outer).
        result = torch.stack(
            [
                torch.stack(_pad_tensors(batch_tensors, N))
                for batch_tensors in tensors
            ]
        )

        return result

def batch_handler(x, handler_fn, **kwargs):
    """Light-weight batch handling function.

    The purpose of this function is to simplify the handling of batches
    of input data, in particular for modules that deal with point cloud
    data. The handler essentially checks whether a 2D array (matrix) or
    a 3D array (tensor) was provided, and calls a handler function. The
    idea of the handler function is to handle an individual 2D array.

    Parameters
    ----------
    x : array_like
        Input point cloud(s). Can be either 2D array, indicating
        a single point cloud, or a 3D array, or even a *list* of
        point clouds (of potentially different cardinalities).

    handler_fn : callable
        Function to call for handling a 2D array.

    **kwargs
        Additional arguments to provide to `handler_fn`.

    Returns
    -------
    list or individual value
        Depending on whether `x` needs to be unwrapped, this function
        returns either a single value or a list of values, resulting
        from calling `handler_fn` on individual parts of `x`.
    """
    # Check whether individual batches need to be handled (3D array)
    # or not (2D array). We default to this type of processing for a
    # list as well.
    if isinstance(x, list) or len(x.shape) == 3:
        # TODO: This type of batch handling is rather ugly and
        # inefficient but at the same time, it is the easiest
        # workaround for now, permitting even 'ragged' inputs of
        # different lengths.
        return [handler_fn(torch.as_tensor(x_), **kwargs) for x_ in x]
    else:
        return handler_fn(torch.as_tensor(x), **kwargs)

class PersistenceInformation(
    namedtuple(
        "PersistenceInformation",
        [
            "pairing",
            "diagram",
            "dimension",
        ],
        # Ensures that there is always a dimension specified, albeit an
        # 'incorrect' one.
        defaults=[None],
    )
):
    """Persistence information data structure.

    This is a light-weight data structure for carrying information about
    the calculation of persistent homology. It consists of the following
    components:

    - A *persistence pairing*
    - A *persistence diagram*
    - An (optional) *dimension*

    Due to its lightweight nature, no validity checks are performed, but
    all calculation modules should return a sequence of instances of the
    :class:`PersistenceInformation` class.

    Since this data class is shared with modules that are capable of
    calculating persistent homology, the exact form of the persistence
    pairing might change. Please refer to the respective classes for
    more documentation.
    """

    __slots__ = ()

    # Disable iterating over the class since it collates heterogeneous
    # information and should rather be treated as a building block.
    __iter__ = None

class VietorisRipsComplex(nn.Module):
    """Calculate Vietoris--Rips complex of a data set.

    This module calculates 'differentiable' persistence diagrams for
    point clouds. The underlying topological approximations are done
    by calculating a Vietoris--Rips complex of the data.
    """

    def __init__(
        self,
        dim=1,
        p=2,
        threshold=numpy.inf,
        keep_infinite_features=False,
        **kwargs
    ):
        """Initialise new module.

        Parameters
        ----------
        dim : int
            Calculates persistent homology up to (and including) the
            prescribed dimension.

        p : float
            Exponent indicating which Minkowski `p`-norm to use for the
            calculation of pairwise distances between points. Note that
            if `treat_as_distances` is supplied to :func:`forward`, the
            parameter is ignored and will have no effect. The rationale
            is to permit clients to use a pre-computed distance matrix,
            while always falling back to Minkowski norms.

        threshold : float
            If set to a finite number, only calculates topological
            features up to the specified distance threshold. Thus,
            any persistence pairings may contain infinite features
            as well.

        keep_infinite_features : bool
            If set, keeps infinite features. This flag is disabled by
            default. The rationale for this is that infinite features
            require more deliberate handling and, in case `threshold`
            is not changed, only a *single* infinite feature will not
            be considered in subsequent calculations.

        **kwargs
            Additional arguments to be provided to ``ripser``, i.e. the
            backend for calculating persistent homology. The `n_threads`
            parameter, which controls parallelisation, is probably the
            most relevant parameter to be adjusted.
            Please refer to the `the gitto-ph documentation
            <https://giotto-ai.github.io/giotto-ph/build/html/index.html>`_
            for more details on admissible parameters.

        Notes
        -----
        This module currently only supports Minkowski norms. It does not
        yet support other metrics internally. To use custom metrics, you
        need to set `treat_as_distances` in the :func:`forward` function
        instead.
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.threshold = threshold
        self.keep_infinite_features = keep_infinite_features

        # Ensures that the same parameters are used whenever calling
        # `ripser`.
        self.ripser_params = {
            'return_generators': True,
            'maxdim': self.dim,
            'thresh': self.threshold
        }

        self.ripser_params.update(kwargs)

    def forward(self, x, treat_as_distances=False):
        """Implement forward pass for persistence diagram calculation.

        The forward pass entails calculating persistent homology on
        a point cloud and returning a set of persistence diagrams.

        Parameters
        ----------
        x : array_like
            Input point cloud(s). `x` can either be a 2D array of shape
            `(n, d)`, which is treated as a single point cloud, or a 3D
            array/tensor of the form `(b, n, d)`, with `b` representing
            the batch size. Alternatively, you may also specify a list,
            possibly containing point clouds of non-uniform sizes.

        treat_as_distances : bool
            If set, treats `x` as containing pre-computed distances
            between points. The semantics of how `x` is handled are
            not changed; the only difference is that when `x` has a
            shape of `(n, d)`, the values of `n` and `d` need to be
            the same.

        Returns
        -------
        list of :class:`PersistenceInformation`
            List of :class:`PersistenceInformation`, containing both the
            persistence diagrams and the generators, i.e. the
            *pairings*, of a certain dimension of topological features.
            If `x` is a 3D array, returns a list of lists, in which the
            first dimension denotes the batch and the second dimension
            refers to the individual instances of
            :class:`PersistenceInformation` elements.

            Generators will be represented in the persistence pairing
            based on vertex--edge pairs (dimension 0) or edge--edge
            pairs. Thus, the persistence pairing in dimension zero will
            have three components, corresponding to a vertex and an
            edge, respectively, while the persistence pairing for higher
            dimensions will have four components.
        """
        return batch_handler(
            x,
            self._forward,
            treat_as_distances=treat_as_distances
        )

    def _forward(self, x, treat_as_distances=False):
        """Handle a *single* point cloud.

        This internal function handles the calculation of topological
        features for a single point cloud, i.e. an `array_like` of 2D
        shape.

        Parameters
        ----------
        x : array_like of shape `(n, d)`
            Single input point cloud.

        treat_as_distances : bool
            Flag indicating whether `x` should be treated as a distance
            matrix. See :func:`forward` for more information.

        Returns
        -------
        list of class:`PersistenceInformation`
            List of persistence information data structures, containing
            the persistence diagram and the persistence pairing of some
            dimension in the input data set.
        """
        if treat_as_distances:
            distances = x
        else:
            distances = torch.cdist(x, x, p=self.p)

        generators = ripser_parallel(
            distances.cpu().detach().numpy(),
            metric='precomputed',
            **self.ripser_params
        )['gens']

        # We always have 0D information.
        persistence_information = \
            self._extract_generators_and_diagrams(
                distances,
                generators,
                dim0=True,
            )

        if self.keep_infinite_features:
            persistence_information_inf = \
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    finite=False,
                    dim0=True,
                )

        # Check whether we have any higher-dimensional information that
        # we should return.
        if self.dim >= 1:
            persistence_information.extend(
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    dim0=False,
                )
            )

            if self.keep_infinite_features:
                persistence_information_inf.extend(
                    self._extract_generators_and_diagrams(
                        distances,
                        generators,
                        finite=False,
                        dim0=False,
                    )
                )

        # Concatenation is only necessary if we want to keep infinite
        # features.
        if self.keep_infinite_features:
            persistence_information = self._concatenate_features(
                persistence_information, persistence_information_inf
            )

        return persistence_information

    def _extract_generators_and_diagrams(
            self,
            dist,
            gens,
            finite=True,
            dim0=False
    ):
        """Extract generators and persistence diagrams from raw data.

        This convenience function translates between the output of
        `ripser_parallel` and the required output of this function.
        """
        index = 1 if not dim0 else 0

        # Perform index shift to find infinite features in the tensor.
        if not finite:
            index += 2

        gens = gens[index]

        if dim0:
            if finite:
                # In a Vietoris--Rips complex, all vertices are created at
                # time zero.
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:, 0],
                    device=dist.device
                )

                destroyers = dist[gens[:, 1], gens[:, 2]]
            else:
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:],
                    device=dist.device
                )

                destroyers = torch.full_like(
                    torch.as_tensor(gens)[:],
                    torch.inf,
                    dtype=torch.float,
                    device=dist.device
                )

                inf_pairs = numpy.full(
                    shape=(gens.shape[0], 2), fill_value=-1
                )
                gens = numpy.column_stack((gens, inf_pairs))

            persistence_diagram = torch.stack(
                (creators, destroyers), 1
            )

            return [PersistenceInformation(gens, persistence_diagram, 0)]
        else:
            result = []

            for index, gens_ in enumerate(gens):
                # Dimension zero is handled differently, so we need to
                # use an offset here. Note that this is not used as an
                # index into the `gens` array any more.
                dimension = index + 1

                if finite:
                    creators = dist[gens_[:, 0], gens_[:, 1]]
                    destroyers = dist[gens_[:, 2], gens_[:, 3]]

                    persistence_diagram = torch.stack(
                        (creators, destroyers), 1
                    )
                else:
                    creators = dist[gens_[:, 0], gens_[:, 1]]

                    destroyers = torch.full_like(
                        torch.as_tensor(gens_)[:, 0],
                        torch.inf,
                        dtype=torch.float,
                        device=dist.device
                    )

                    # Create special infinite pairs; we pretend that we
                    # are concatenating with unknown edges here.
                    inf_pairs = numpy.full(
                        shape=(gens_.shape[0], 2), fill_value=-1
                    )
                    gens_ = numpy.column_stack((gens_, inf_pairs))

                persistence_diagram = torch.stack(
                    (creators, destroyers), 1
                )

                result.append(
                    PersistenceInformation(
                        gens_,
                        persistence_diagram,
                        dimension)
                )

        return result

    def _concatenate_features(self, pers_info_finite, pers_info_infinite):
        """Concatenate finite and infinite features."""
        def _apply(fin, inf):
            assert fin.dimension == inf.dimension

            diagram = torch.concat((fin.diagram, inf.diagram))
            pairing = numpy.concatenate((fin.pairing, inf.pairing), axis=0)
            dimension = fin.dimension

            return PersistenceInformation(
                pairing=pairing,
                diagram=diagram,
                dimension=dimension
            )

        return list(starmap(_apply, zip(pers_info_finite, pers_info_infinite)))

class StructureElementLayer(torch.nn.Module):
    def __init__(
        self,
        n_elements
    ):
        super().__init__()

        self.n_elements = n_elements
        self.dim = 2    # TODO: Make configurable

        size = (self.n_elements, self.dim)

        self.centres = torch.nn.Parameter(
            torch.rand(*size)
        )

        self.sharpness = torch.nn.Parameter(
            torch.ones(*size) * 3
        )

    def forward(self, x):
        batch = torch.cat([x] * self.n_elements, 1)

        B, N, D = x.shape

        # This is a 'butchered' variant of the much nicer `SLayerExponential`
        # class by C. Hofer and R. Kwitt.
        #
        # https://c-hofer.github.io/torchph/_modules/torchph/nn/slayer.html#SLayerExponential

        centres = torch.cat([self.centres] * N, 1)
        centres = centres.view(-1, self.dim)
        centres = torch.stack([centres] * B, 0)
        centres = torch.cat((centres, 2 * batch[..., -1].unsqueeze(-1)), 2)

        sharpness = torch.pow(self.sharpness, 2)
        sharpness = torch.cat([sharpness] * N, 1)
        sharpness = sharpness.view(-1, self.dim)
        sharpness = torch.stack([sharpness] * B, 0)
        sharpness = torch.cat(
            (
                sharpness,
                torch.ones_like(batch[..., -1].unsqueeze(-1))
            ),
            2
        )

        x = centres - batch
        x = x.pow(2)
        x = torch.mul(x, sharpness)
        x = torch.nansum(x, 2)
        x = torch.exp(-x)
        x = x.view(B, self.n_elements, -1)
        x = torch.sum(x, 2)
        x = x.squeeze()

        return x