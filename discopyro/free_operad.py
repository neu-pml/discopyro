import collections
import functools
import itertools

from discopy.closed import Under
from discopy.monoidal import Box, Category, Diagram, Functor, Id, PRO, Ty
from discopy.python import Function
import discopy.wiring as wiring
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}',
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
    }
)


import networkx as nx
import os.path
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import pyvis
import pyvis.network
import re
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F

from . import unification, util

NONE_DEFAULT = collections.defaultdict(lambda: None)

def functionize(f):
    """Transform a :class:`discopy.monoidal.Box` into a
       :class:`discopy.python.Function`

    :param f: A box with a callable function as data
    :type f: :class:`Box`
    """
    return Function(f.data['function'], PRO(len(f.dom)), PRO(len(f.cod)))

_PYTHON_FUNCTOR = Functor(
    ob=lambda t: PRO(len(t)), ar=functionize,
    cod=Category(PRO, Function)
)

Diagram.__call__ = lambda self, *values: _PYTHON_FUNCTOR(self)(*values)

class FreeOperad(pyro.nn.PyroModule):
    """Pyro module representing a free operad as a graph, and implementing
    stochastic shortest-paths sampling of operations.

    :param list generators: A set of :class:`discopy.monoidal.Box` types
                            representing the generating operations of the free
                            operad.
    :param list global_elements: A set of :class:`discopy.monoidal.Box`
                                 representing the (often stochastic) global
                                 elements of the free operad's types
    """
    def __init__(self, generators, global_elements):
        super().__init__()
        self._graph = nx.DiGraph()
        self._generators = collections.defaultdict(list)
        self._generator_indices = {}

        self._add_type(Ty())
        for i, gen in enumerate(generators):
            self._add_generator(gen, name='generator_%d' % i)

        for i, obj in enumerate(self.obs):
            self._graph.nodes[obj]['type_index'] = i

        for i, elem in enumerate(global_elements):
            assert unification.equiv(elem.dom, Ty())
            self._add_generator(elem, name='global_element_%d' % i)

        stack = set(self.compound_obs)
        i = 0
        while stack:
            ty = stack.pop()
            if isinstance(ty, Under):
                diagram = wiring.Box('', obj.left, obj.right, data={})
                self._add_macro(diagram, Ty(), obj)
            else:
                chunk_inhabitants = [{(dom, chunk) for (dom, cod) in
                                      self._skeleton_generators(chunk, False)}
                                     for chunk in self._chunk_type(ty)]
                for tensor in itertools.product(*chunk_inhabitants):
                    boxes = [wiring.Box('', dom, cod) for dom, cod in tensor]
                    diagram = functools.reduce(lambda f, g: f @ g, boxes,
                                               wiring.Id(Ty()))
                    if util.type_contains(ty, diagram.dom):
                        continue
                    if diagram.dom not in self._graph:
                        stack.add(diagram.dom)
                    self._add_macro(diagram, diagram.dom, diagram.cod)

        self.arrow_weight_alpha = pnn.PyroParam(
            torch.ones(len(self.ars) + len(self.macros)),
            constraint=constraints.positive
        )
        self.temperature_alpha = pnn.PyroParam(torch.ones(1),
                                               constraint=constraints.positive)
        self.temperature_beta = pnn.PyroParam(torch.ones(1),
                                              constraint=constraints.positive)

        self.register_buffer('adjacency',
                             torch.from_numpy(nx.to_numpy_array(self._graph)),
                             persistent=False)
        adjacency = torch.clone(self.adjacency)
        self.register_buffer('diffusions', adjacency.matrix_exp(),
                             persistent=False)

    def _dom(self, arrow):
        if isinstance(arrow, Ty):
            return arrow
        return list(self._graph.in_edges(arrow))[0][0]

    def _cod(self, arrow):
        if isinstance(arrow, Ty):
            return arrow
        return list(self._graph.out_edges(arrow))[0][1]

    def _add_generator(self, gen, name='generator'):
        assert isinstance(gen, Box)
        self._add_type(gen.dom)
        self._add_type(gen.cod)
        homset = (gen.dom, gen.cod)

        if gen not in self._generators[homset]:
            self._generator_indices[gen] =\
                len(list(itertools.chain(*self._generators.values())))
            self._generators[homset].append(gen)
            if isinstance(gen.data['function'], nn.Module):
                self.add_module(name, gen.data['function'])

            if homset not in self._graph:
                self._graph.add_node(homset, index=len(self._graph))
                self._graph.add_edge(gen.dom, homset)
                self._graph.add_edge(homset, gen.cod)

    def _add_type(self, obj):
        """Add an type as a node to the graph representing the free operad

        :param obj: A type representing an type in the free operad
        :type obj: :class:`discopy.monoidal.Ty`
        """
        if obj in self._graph:
            return
        if unification.type_compound(obj):
            if len(obj) > 1:
                for ob in obj:
                    self._add_type(Ty(ob))
            else:
                dom, cod = obj.left, obj.right
                self._add_type(dom)
                self._add_type(cod)
        self._graph.add_node(obj, index=len(self._graph))

    def _add_macro(self, macro, dom, cod):
        assert isinstance(macro, wiring.Diagram)
        self._add_type(dom)
        self._add_type(cod)
        homset = (dom, cod)

        if macro not in self._generators[homset]:
            self._generator_indices[macro] =\
                len(list(itertools.chain(*self._generators.values())))
            self._generators[homset].append(macro)

            if homset not in self._graph:
                self._graph.add_node(homset, index=len(self._graph))
                self._graph.add_edge(dom, homset)
                self._graph.add_edge(homset, cod)

    def _node_index(self, node):
        return self._graph.nodes[node]['index']

    def _chunk_type(self, ty):
        chunking = []
        remainder = ty
        while len(remainder):
            i = max((i for i in range(1, len(remainder))
                     if remainder[:i] in self._graph), default=1)
            chunking.append(remainder[:i])
            remainder = remainder[i:]
        return chunking

    @property
    def param_shapes(self):
        """Parameter shapes that must be provided to sample

        :returns: Tuple containing the shape of arrow weights parameters (for a
                  Dirichlet distribution) and temperature parameters (for a Beta
                  distribution)
        :rtype: tuple
        """
        return (self.arrow_weight_alpha.shape, self.temperature_alpha.shape * 2)

    def _skeleton_arrows(self, ty, forward=True):
        """Return a list of skeleton edges flowing into or out of an type

        :param ty: The type into or out of which arrows flow
        :type ty: :class:`discopy.monoidal.Ty`

        :param bool forward: Whether to look for forwards (True) or backwards
                             (False) paths between the types

        :returns: Skeleton edges into or out of `ty`
        :rtype: generator
        """
        if forward:
            edges, direction = self._graph.out_edges(ty), 1
        else:
            edges, direction = self._graph.in_edges(ty), 0

        for edge in edges:
            yield edge[direction]

    def _skeleton_generators(self, src, forward=True):
        for (dom, cod) in self._skeleton_arrows(src, forward):
            if any(isinstance(g, Box) for g in self._generators[(dom, cod)]):
                yield (dom, cod)

    def _skeleton_bridges(self, src, dest, forward=True):
        """Return a list of skeleton edges bridging one type to another

        :param src: The source from which to walk
        :type dest: :class:`discopy.monoidal.Ty`
        :param dest: The destination to which to walk
        :type dest: :class:`discopy.monoidal.Ty`

        :param bool forward: Whether to look for forwards (True) or backwards
                             (False) paths between the types

        :returns: Skeleton edges able to connect `src` to `dest`
        :rtype: generator
        """
        for (dom, cod) in self._skeleton_arrows(src, forward):
            if forward:
                i, j = self._node_index(cod), self._node_index(dest)
            else:
                i, j = self._node_index(dest), self._node_index(dom)

            if (self.diffusions[i, j] > 0).item():
                yield (dom, cod)

    @property
    def obs(self):
        """A list of the types in the free operad

        :returns: The types in the free operad
        :rtype: list
        """
        return [node for node in self._graph if isinstance(node, Ty)]

    @property
    def compound_obs(self):
        """A list of the compound (product and exponential) types in the free
           operad

        :returns: The product and exponential types in the free operad
        :rtype: list
        """
        for obj in self.obs:
            if unification.type_compound(obj):
                yield obj

    @property
    def ars(self):
        """A list of the generating operations in the free operad

        :returns: The generators in the free operad
        :rtype: list
        """
        return [arrow for arrow in itertools.chain(*self._generators.values())
                if isinstance(arrow, Box)]

    @property
    def macros(self):
        """A list of the generating macros in the free operad

        :returns: The macros in the free operad
        :rtype: list
        """
        return [arrow for arrow in itertools.chain(*self._generators.values())
                if not isinstance(arrow, Box)]

    @pnn.pyro_method
    def hom_arrow(self, hom, path_data, weights, temperature, min_depth=0,
                  infer={}):
        """Sample an arrow from the hom-set from type `dom` to type `cod`.

        :param hom: Hom-set tuple of desired operation's domain and codomain
        :type hom: :class:`tuple[discopy.monoidal.Ty]`

        :param weights: Relative weights for sampling generators at a hom-set
        :type weights: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling
                            operations
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: An operation from `dom` to `cod`
        :rtype: :class:`discopy.monoidal.Diagram`
        """
        if hom not in self._graph:
            raise NotImplementedError()

        indices = torch.tensor([self._generator_indices[g] for g
                                in self._generators[hom]],
                               dtype=torch.long).to(device=temperature.device)

        mask = torch.tensor([1. if isinstance(gen, Box) else (temperature+1e-10)
                             for gen in self._generators[hom]])
        masked_ws = weights[indices] / mask.to(device=temperature.device)

        generator_categorical = dist.Categorical(probs=masked_ws).to_event(0)
        g_idx = pyro.sample('hom(%s, %s)' % hom, generator_categorical,
                            infer=infer)

        generator = self._generators[hom][g_idx.item()]
        if isinstance(generator, Box):
            arrow = generator
        elif isinstance(generator, wiring.Diagram):
            arrow = self.sample_operation(generator, weights, temperature + 1,
                                          min_depth, infer)
        else:
            raise NotImplementedError()

        return arrow, path_data

    @pnn.pyro_method
    def path_through(self, box, weights, temperature, min_depth=0, infer={}):
        """Sample an operation to fill in a wiring diagram box

        :param box: A wiring diagram box of the desired operation
        :type box: :class:`discopy.wiring.Box`

        :param weights: Relative weights for sampling generators at a hom-set
        :type weights: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling
                            operations
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: An operation from `box.dom` to `box.cod`
        :rtype: :class:`discopy.monoidal.Diagram`
        """
        if unification.equiv(box.cod, Ty()) and not box.data:
            return Box(box.name, box.dom, box.cod, data={
                **box.data, 'function': lambda *xs: ()
            })
        dest = self._node_index(box.cod)
        if dist.is_validation_enabled():
            nx.shortest_path(self._graph, box.dom, box.cod)

        location = box.dom
        path = Id(box.dom)
        path_data = box.data
        with pyro.markov():
            while location != box.cod:
                pred = util.HomsetPredicate(len(path), min_depth, box.cod,
                                            self._generators, path_data)

                homs = list(filter(pred, self._skeleton_bridges(location,
                                                                box.cod)))
                bs = torch.tensor(list(map(self._node_index, homs)),
                                  dtype=torch.long)
                bs = bs.to(device=temperature.device)

                logits = self.diffusions[bs, dest].log() / temperature
                homs_categorical = dist.Categorical(logits=logits)
                b_idx = pyro.sample('bridge{%s -> %s, %s}' % (box.dom, box.cod,
                                                              location),
                                    homs_categorical.to_event(0),
                                    infer=infer)

                operation, path_data = self.hom_arrow(homs[b_idx.item()],
                                                      path_data, weights,
                                                      temperature,
                                                      min_depth - len(path) - 1,
                                                      infer)
                path = path >> operation
                location = operation.cod

        return path

    @pnn.pyro_method
    def sample_operation(self, diagram, weights, temperature, min_depth=0,
                         infer={}):
        """Sample an operation from the terminal type into a specified type

        :param obj: Target type
        :type obj: :class:`discopy.monoidal.Ty`
        :param temperature: Temperature (scale parameter) for sampling operations
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: An operation from Ty() to `obj`
        :rtype: :class:`discopy.monoidal.Diagram`
        """

        with name_count():
            functor = wiring.Functor(lambda t: t,
                                     lambda f: self.path_through(f, weights,
                                                                 temperature,
                                                                 min_depth,
                                                                 infer),
                                     cod=Category(Ty, Box))
            return functor(diagram)

    def forward(self, diagram, min_depth=0, infer={}, temperature=None,
                arrow_weights=None):
        """Sample an operation from the terminal type into a specified type

        :param obj: Target type
        :type obj: :class:`discopy.monoidal.Ty`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :param temperature: Temperature (scale parameter) for sampling operations
        :type temperature: :class:`torch.Tensor`

        :param arrow_weights: Proposed arrow weights to combine with the
                              long-run arrival probabilities
        :type arrow_weights: :class:`torch.Tensor`

        :returns: An operation from Ty() to `obj`
        :rtype: :class:`discopy.monoidal.Diagram`
        """
        if temperature is None:
            temperature = pyro.sample(
                'weights_temperature',
                dist.Gamma(self.temperature_alpha,
                           self.temperature_beta).to_event(0)
            )
        if arrow_weights is None:
            arrow_weights = pyro.sample(
                'arrow_weights',
                dist.Dirichlet(self.arrow_weight_alpha)
            )

        return self.sample_operation(diagram, arrow_weights, temperature,
                                     min_depth, infer)

    def __reachability_falg__(self, diagram):
        if isinstance(diagram, wiring.Box):
            return nx.has_path(self._graph, diagram.dom, diagram.cod)
        if isinstance(diagram, wiring.Id):
            return True
        if isinstance(diagram, wiring.Sequential):
            return all(diagram.arrows)
        if isinstance(diagram, wiring.Parallel):
            return all(diagram.factors)

    def reachable(self, diagram):
        return diagram.collapse(self.__reachability_falg__)

    def skeleton(self, skip_edges=[]):
        """Construct the skeleton graph for the underlying free operad

        :param list skip_edges: List of arrows to skip

        :returns: Skeleton graph
        :rtype: :class:`nx.Digraph`
        """
        skeleton = nx.DiGraph()

        for node in self._graph.nodes:
            skeleton.add_node(util.node_name(node))

        for u, v in self._graph.edges:
            if (u, v) in skip_edges:
                continue
            skeleton.add_edge(util.node_name(u), util.node_name(v))
        return skeleton

    def draw(self, skip_edges=[], filename=None, notebook=False):
        """Draw the free operad's nerve/skeleton using either PyVis or
           networkX

        :param list skip_edges: List of arrows to skip
        :param str filename: Filename to which to save the figure
        :param bool notebook: Whether the caller is within a Jupyter notebook
        """
        skeleton = self.skeleton(skip_edges)

        if filename and os.path.splitext(filename)[1] == '.html':
            vis = pyvis.network.Network(layout='hierarchical', directed=True)
            vis.options.layout.hierarchical.direction = 'LR'
            vis.options.layout.hierarchical.sortMethod = 'directed'

            vis.from_nx(skeleton)
            vis.toggle_physics(True)
            vis.save_graph(filename)
            if notebook:
                from IPython.core.display import display, HTML
                display(HTML(filename))
        else:
            pos = nx.spring_layout(skeleton, k=10, weight='weight')
            nx.draw_networkx_nodes(skeleton, pos, node_size=700)
            nx.draw_networkx_edges(skeleton, pos, node_size=700, width=2,
                                   edge_color='gray', alpha=0.75)
            nx.draw_networkx_labels(skeleton, pos, font_size=12)
            plt.axis("off")

            if filename:
                plt.savefig(filename)
            plt.show()

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])
