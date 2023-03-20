import collections
import functools
import itertools

from discopy.closed import Under
from discopy.monoidal import Box, Category, Id, Ty
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

from . import dagger, unification, util

NONE_DEFAULT = collections.defaultdict(lambda: None)

class FreeOperad(pyro.nn.PyroModule):
    """Pyro module representing a free operad as a graph, and implementing
    stochastic shortest-paths sampling of operations.

    :param list generators: A set of :class:`discopy.biclosed.Box` types
                            representing the generating operations of the free
                            operad.
    :param list global_elements: A set of :class:`discopy.biclosed.Box`
                                 representing the (often stochastic) global
                                 elements of the free operad's types
    """
    def __init__(self, generators, global_elements):
        super().__init__()
        self._graph = nx.DiGraph()
        self._add_type(Ty())
        for i, gen in enumerate(generators):
            self._add_generator(gen, i, name='generator_%d' % i)

        for i, obj in enumerate(self.obs):
            self._graph.nodes[obj]['type_index'] = i

        for i, elem in enumerate(global_elements):
            assert unification.equiv(elem.dom, Ty())
            self._add_generator(elem, len(generators) + 1,
                                name='global_element_%d' % i)

        stack = set(self.compound_obs)
        i = 0
        while stack:
            ty = stack.pop()
            if isinstance(ty, Under):
                diagram = wiring.Box('', obj.left, obj.right, data={})
                self._add_macro(diagram, Ty(), obj,
                                len(generators) + len(global_elements) + i)
            else:
                chunk_inhabitants = [{(ar.dom, chunk) for (ar, _, _, _)
                                      in self._type_generators(chunk, False)
                                      if isinstance(ar, Box)}
                                     for chunk in self._chunk_type(ty)]
                for tensor in itertools.product(*chunk_inhabitants):
                    boxes = [wiring.Box('', dom, cod) for dom, cod in tensor]
                    diagram = functools.reduce(lambda f, g: f @ g, boxes,
                                               wiring.Id(Ty()))
                    if util.type_contains(ty, diagram.dom):
                        continue
                    if diagram.dom not in self._graph:
                        stack.add(diagram.dom)
                    self._add_macro(diagram, diagram.dom, diagram.cod,
                                    len(generators) + len(global_elements) + i)

        self.arrow_weight_loc = pnn.PyroParam(
            torch.zeros(len(self.ars) + len(self.macros)),
        )
        self.arrow_weight_scale = pnn.PyroParam(
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
        adjacency_weights = torch.clone(self.adjacency).detach()
        arrow_indices = [self._index(arrow) for arrow in self.ars + self.macros]
        self.register_buffer('arrow_indices', torch.tensor(arrow_indices,
                                                           dtype=torch.long),
                             persistent=False)
        self.register_buffer('diffusions', adjacency_weights.matrix_exp(),
                             persistent=False)

    def _dom(self, arrow):
        if isinstance(arrow, Ty):
            return arrow
        return list(self._graph.in_edges(arrow))[0][0]

    def _cod(self, arrow):
        if isinstance(arrow, Ty):
            return arrow
        return list(self._graph.out_edges(arrow))[0][1]

    def _add_generator(self, gen, arrow_index, name='generator'):
        assert isinstance(gen, Box)
        self._add_type(gen.dom)
        self._add_type(gen.cod)

        if gen not in self._graph:
            self._graph.add_node(gen, index=len(self._graph),
                                 arrow_index=arrow_index)
            self._graph.add_edge(gen.dom, gen)
            self._graph.add_edge(gen, gen.cod)

            if isinstance(gen.data['function'], nn.Module):
                self.add_module(name, gen.data['function'])
            if isinstance(gen, dagger.DaggerBox):
                dag = gen.dagger()
                if isinstance(dag.data['function'], nn.Module):
                    self.add_module(name + '_dagger', dag.data['function'])

    def _add_type(self, obj):
        """Add an type as a node to the graph representing the free operad

        :param obj: A type representing an type in the free operad
        :type obj: :class:`discopy.biclosed.Ty`
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

    def _add_macro(self, macro, dom, cod, arrow_index):
        assert isinstance(macro, wiring.Diagram)
        self._add_type(dom)
        self._add_type(cod)

        if not macro in self._graph:
            self._graph.add_node(macro, index=len(self._graph),
                                 arrow_index=arrow_index)
            self._graph.add_edge(dom, macro)
            self._graph.add_edge(macro, cod)

    def _index(self, node, arrow=False):
        key = 'arrow_index' if arrow else 'index'
        return self._graph.nodes[node][key]

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

        :returns: Tuple containing the shape of arrow weights and temperature
                  parameters (for a Beta distribution)
        :rtype: tuple
        """
        return (self.arrow_weights.shape, self.temperature_alpha.shape * 2)

    def _type_generators(self, obj, forward=True, pred=None):
        """Return a list of generators flowing into or out of an type

        :param obj: The type in question
        :type obj: :class:`discopy.biclosed.Ty`
        :param bool forward: Whether to look for arrows flowing out from (True)
                             or into (False) an type

        :returns: A list of generating operations connected to `obj`
        :rtype: list
        """
        edges = self._graph.out_edges if forward else self._graph.in_edges
        dir_index = 1 if forward else 0
        for edge in edges(obj):
            gen = edge[dir_index]
            cod = list(edges(gen))[0][dir_index]

            if pred:
                cod_index, dest_index = self._index(cod), self._index(pred.cod)
                connected = (self.diffusions[cod_index, dest_index] > 0).item()
            else:
                connected = True

            if connected and (pred is None or pred(gen, cod)):
                yield (gen, cod, self._index(gen), self._index(gen, True))

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
        return [node for node in self._graph
                if isinstance(node, Box)]

    @property
    def macros(self):
        """A list of the generating macros in the free operad

        :returns: The macros in the free operad
        :rtype: list
        """
        return [node for node in self._graph if not isinstance(node, Ty) and\
                not isinstance(node, Box)]

    @pnn.pyro_method
    def weights_matrix(self, arrow_weights):
        """Construct the matrix of transition weights between nodes in the
           free operad's graph representation, given weights for the arrows.

        :param arrow_weights: Weights for the arrows, indexed by arrow indices
        :type arrow_weights: :class:`torch.Tensor`

        :returns: An unnormalized transition matrix for a random walk over the
                  nerve of the free operad.
        :rtype: :class:`torch.Tensor`
        """
        weights = arrow_weights.unsqueeze(dim=-1)
        weights = self.adjacency[self.arrow_indices, :] * weights
        return self.adjacency.index_put((self.arrow_indices,), weights)

    @pnn.pyro_method
    def path_through(self, box, energies, temperature, min_depth=0, infer={}):
        """Sample an operation from type `src` to type `dest_mask`

        :param src: Source type, the desired operation's domain
        :type src: :class:`discopy.biclosed.Ty`
        :param dest_mask: Destination type, the desired operation's codomain
        :type dest_mask: :class:`discopy.biclosed.Ty`

        :param energies: Matrix of long-run arrival probabilities in the graph
        :type energies: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling operations
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: An operation from `src` to `dest_mask`
        :rtype: :class:`discopy.biclosed.Diagram`
        """
        if unification.equiv(box.cod, Ty()) and not box.data:
            return Box(box.name, box.dom, box.cod, lambda *xs: (),
                                   data=box.data)
        dest = self._index(box.cod)
        if dist.is_validation_enabled():
            nx.shortest_path(self._graph, box.dom, box.cod)

        location = box.dom
        path = Id(box.dom)
        path_data = box.data
        with pyro.markov():
            while location != box.cod or not path.inside:
                pred = util.GeneratorPredicate(len(path), min_depth, path_data,
                                               box.cod)
                generators = list(self._type_generators(location, True, pred))
                gens = torch.tensor([g for (_, _, g, _) in generators],
                                    dtype=torch.long).to(device=energies.device)

                logits = energies[gens, dest] / (temperature + 1e-10)
                generators_categorical = dist.Categorical(logits=logits)
                g_idx = pyro.sample('path_step{%s -> %s, %s}' % (box.dom,
                                                                 box.cod,
                                                                 location),
                                    generators_categorical.to_event(0),
                                    infer=infer)

                gen, cod, _, _ = generators[g_idx.item()]
                if isinstance(gen, Box):
                    operation = gen
                    if gen.data and path_data:
                        updated_data = {**path_data}
                        for k, v in path_data.items():
                            if not callable(v):
                                next_v = v[1:]
                                if next_v:
                                    updated_data[k] = next_v
                                else:
                                    del updated_data[k]

                        path_data = updated_data
                elif isinstance(gen, wiring.Diagram):
                    operation = self.sample_operation(gen, energies,
                                                      temperature,
                                                      min_depth - len(path) - 1,
                                                      infer)
                else:
                    raise NotImplementedError()
                path = path >> operation
                location = cod

        return path

    @pnn.pyro_method
    def sample_operation(self, diagram, energies, temperature, min_depth=0,
                         infer={}):
        """Sample an operation from the terminal type into a specified type

        :param obj: Target type
        :type obj: :class:`discopy.biclosed.Ty`
        :param energies: Matrix of long-run arrival probabilities in the graph
        :type energies: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling operations
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: An operation from Ty() to `obj`
        :rtype: :class:`discopy.biclosed.Diagram`
        """

        with name_count():
            functor = wiring.Functor(lambda t: t,
                                     lambda f: self.path_through(f, energies,
                                                                 temperature,
                                                                 min_depth,
                                                                 infer),
                                     cod=Category(Ty, Box))
            return functor(diagram)

    def forward(self, diagram, min_depth=0, infer={}, temperature=None,
                arrow_weights=None):
        """Sample an operation from the terminal type into a specified type

        :param obj: Target type
        :type obj: :class:`discopy.biclosed.Ty`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :param temperature: Temperature (scale parameter) for sampling operations
        :type temperature: :class:`torch.Tensor`

        :param arrow_weights: Proposed arrow weights to combine with the
                              long-run arrival probabilities
        :type arrow_weights: :class:`torch.Tensor`

        :returns: An operation from Ty() to `obj`
        :rtype: :class:`discopy.biclosed.Diagram`
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
                dist.Normal(self.arrow_weight_loc,
                            self.arrow_weight_scale).to_event(1)
            )

        weights = self.diffusions + self.weights_matrix(arrow_weights)
        return self.sample_operation(diagram, weights, temperature, min_depth,
                                     infer)

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
        arrow_weights = dist.Normal(self.arrow_weight_loc,
                                    self.arrow_weight_scale).mean

        skeleton = nx.DiGraph()

        for node in self._graph.nodes:
            props = {}
            if 'arrow_index' in self._graph.nodes[node]:
                k = self._graph.nodes[node]['arrow_index']
                props['weight'] = arrow_weights[k].item()
            skeleton.add_node(util.node_name(node), **props)

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
