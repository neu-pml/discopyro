import collections
from discopy.biclosed import Id, Ty, Under
import functools
import matplotlib.pyplot as plt
import networkx as nx
import os.path
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import pyvis
import pyvis.network
import scipy.linalg
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F

from . import callable, unification

NONE_DEFAULT = collections.defaultdict(lambda: None)

class FreeCategory(pyro.nn.PyroModule):
    """Pyro module representing a free category as a graph, and implementing
    stochastic shortest-paths sampling of morphisms.

    :param list generators: A set of :class:`discopy.biclosed.Box` objects
                            representing the generating morphisms of the free
                            category.
    :param list global_elements: A set of :class:`discopy.biclosed.Box`
                                 representing the (often stochastic) global
                                 elements of the free category's objects
    """
    def __init__(self, generators, global_elements):
        super().__init__()
        self._graph = nx.DiGraph()
        self._add_object(Ty())
        for i, gen in enumerate(generators):
            assert isinstance(gen, callable.CallableBox)

            if gen.dom not in self._graph:
                self._add_object(gen.dom)
            self._graph.add_node(gen, index=len(self._graph), arrow_index=i)
            if gen.cod not in self._graph:
                self._add_object(gen.cod)
            self._graph.add_edge(gen.dom, gen)
            self._graph.add_edge(gen, gen.cod)

            if isinstance(gen.function, nn.Module):
                self.add_module('generator_%d' % i, gen.function)
            if isinstance(gen, callable.CallableDaggerBox):
                dagger = gen.dagger()
                if isinstance(dagger.function, nn.Module):
                    self.add_module('generator_%d_dagger' % i, dagger.function)

        for i, obj in enumerate(self.obs):
            self._graph.nodes[obj]['object_index'] = i

        for i, elem in enumerate(global_elements):
            assert isinstance(elem, callable.CallableBox)
            assert elem.dom == Ty()
            if not isinstance(elem, callable.CallableDaggerBox):
                dagger_name = '%s$^{\\dagger}$' % elem.name
                elem = callable.CallableDaggerBox(elem.name, elem.dom, elem.cod,
                                                  elem.function,
                                                  lambda *args: (), dagger_name)

            self._graph.add_node(elem, index=len(self._graph),
                                 arrow_index=len(generators) + i)
            self._graph.add_edge(Ty(), elem)
            self._graph.add_edge(elem, elem.cod)

            if isinstance(elem.function, nn.Module):
                self.add_module('global_element_%d' % i, elem.function)
            dagger = elem.dagger()
            if isinstance(dagger.function, nn.Module):
                self.add_module('global_element_%d_dagger' % i, dagger.function)

        for i, obj in enumerate(self.compound_obs):
            if isinstance(obj, Under):
                src, dest = obj.left, obj.right
                def macro(probs, temp, min_depth, infer, l=src, r=dest):
                    return self.path_between(l, r, probs, temp, min_depth,
                                             infer)
            else:
                def macro(probs, temp, min_depth, infer, obj=obj):
                    return self.product_arrow(obj, probs, temp, min_depth,
                                              infer)

            arrow_index = len(generators) + len(global_elements) + i
            self._graph.add_node(macro, index=len(self._graph),
                                 arrow_index=arrow_index)
            self._graph.add_edge(Ty(), macro)
            self._graph.add_edge(macro, obj)

        self.arrow_weight_alphas = pnn.PyroParam(
            torch.ones(len(self.ars) + len(self.macros)),
            constraint=constraints.positive
        )
        self.arrow_weight_betas = pnn.PyroParam(
            torch.ones(len(self.ars) + len(self.macros)),
            constraint=constraints.positive
        )
        self.temperature_alpha = pnn.PyroParam(torch.ones(1),
                                               constraint=constraints.positive)
        self.temperature_beta = pnn.PyroParam(torch.ones(1),
                                              constraint=constraints.positive)

        adjacency_weights = torch.from_numpy(nx.to_numpy_matrix(self._graph))
        for arrow in self.ars:
            i = self._index(arrow)
            adjacency_weights[i] /= self._arrow_parameters(arrow) + 1
        self.register_buffer('diffusion_counts', adjacency_weights.matrix_exp(),
                             persistent=False)

    def _arrow_parameters(self, arrow):
        """Count up the number of parameters a generating morphism has, if its
           implementing function happens to be a :class:`torch.nn.Module`

           :param arrow: A generating morphism within the free category

           :returns: Number of parameters in the arrow (including its dagger)
           :rtype: int
        """
        params = 0
        if isinstance(arrow.function, nn.Module):
            for parameter in arrow.function.parameters():
                params += parameter.numel()
        if isinstance(arrow, callable.CallableDaggerBox):
            dagger = arrow.dagger()
            if isinstance(dagger.function, nn.Module):
                for parameter in dagger.function.parameters():
                    params += parameter.numel()
        return params

    def _add_object(self, obj):
        """Add an object as a node to the graph representing the free category

        :param obj: A type representing an object in the free category
        :type obj: :class:`discopy.biclosed.Ty`
        """
        if obj in self._graph:
            return
        if unification.type_compound(obj):
            if len(obj) > 1:
                for ob in obj:
                    self._add_object(Ty(ob))
            else:
                dom, cod = obj.left, obj.right
                self._add_object(dom)
                self._add_object(cod)
        self._graph.add_node(obj, index=len(self._graph))

    def _index(self, node, arrow=False):
        key = 'arrow_index' if arrow else 'index'
        return self._graph.nodes[node][key]

    @property
    def param_shapes(self):
        """Parameter shapes that must be provided to sample

        :returns: Tuple containing the shape of arrow weights and temperature
                  parameters (for a Beta distribution)
        :rtype: tuple
        """
        return (self.arrow_weights.shape, self.temperature_alpha.shape * 2)

    def _object_generators(self, obj, forward=True):
        """Return a list of generators flowing into or out of an object

        :param obj: The object in question
        :type obj: :class:`discopy.biclosed.Ty`
        :param bool forward: Whether to look for arrows flowing out from (True)
                             or into (False) an object

        :returns: A list of generating morphisms connected to `obj`
        :rtype: list
        """
        edges = self._graph.out_edges if forward else self._graph.in_edges
        dir_index = 1 if forward else 0
        generators = []
        for edge in edges(obj):
            gen = edge[dir_index]
            counterpart = list(edges(gen))[0][dir_index]
            generators.append((gen, counterpart))
        return generators

    @property
    def obs(self):
        """A list of the objects in the free category

        :returns: The objects in the free category
        :rtype: list
        """
        return [node for node in self._graph if isinstance(node, Ty)]

    @property
    def compound_obs(self):
        """A list of the compound (product and exponential) objects in the free
           category

        :returns: The product and exponential objects in the free category
        :rtype: list
        """
        for obj in self.obs:
            if unification.type_compound(obj):
                yield obj

    @property
    def ars(self):
        """A list of the generating morphisms in the free category

        :returns: The generators in the free category
        :rtype: list
        """
        return [node for node in self._graph
                if isinstance(node, callable.CallableBox)]

    @property
    def macros(self):
        """A list of the generating macros in the free category

        :returns: The macros in the free category
        :rtype: list
        """
        return [node for node in self._graph if not isinstance(node, Ty) and\
                not isinstance(node, callable.CallableBox)]

    @pnn.pyro_method
    def weights_matrix(self, arrow_weights):
        """Construct the matrix of transition weights between nodes in the
           free category's graph representation, given weights for the arrows.

        :param arrow_weights: Weights for the arrows, indexed by arrow indices
        :type arrow_weights: :class:`torch.Tensor`

        :returns: An unnormalized transition matrix for a random walk over the
                  nerve of the free category.
        :rtype: :class:`torch.Tensor`
        """
        weights = torch.from_numpy(nx.to_numpy_matrix(self._graph)).to(
            arrow_weights
        )

        for arrow in self.ars:
            i = self._index(arrow)
            k = self._index(arrow, arrow=True)
            weights = weights.index_put((torch.LongTensor([i]),),
                                        weights[i] * arrow_weights[k])

        for macro in self.macros:
            i = self._index(macro)
            k = self._index(macro, arrow=True)
            weights = weights.index_put((torch.LongTensor([i]),),
                                        weights[i] * arrow_weights[k])

        return weights

    @pnn.pyro_method
    def product_arrow(self, obj, probs, temperature, min_depth=0, infer={}):
        """Sample a morphism from the terminal object into a Cartesian product

        :param obj: Cartesian product object to target
        :type obj: :class:`discopy.biclosed.Ty`
        :param probs: Matrix of long-run arrival probabilities in the graph
        :type probs: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling morphisms
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: A morphism from Ty() to `obj`
        :rtype: :class:`discopy.biclosed.Diagram`
        """
        product = None
        for ob in obj.objects:
            entry = self.sample_morphism(Ty(ob), probs, temperature + 1,
                                         min_depth, infer)
            if product is None:
                product = entry
            else:
                product = product @ entry
        return product

    @pnn.pyro_method
    def path_between(self, src, dest, probs, temperature, min_depth=0,
                     infer={}):
        """Sample a morphism from object `src` to object `dest`

        :param src: Source object, the desired morphism's domain
        :type src: :class:`discopy.biclosed.Ty`
        :param dest: Destination object, the desired morphism's codomain
        :type dest: :class:`discopy.biclosed.Ty`

        :param probs: Matrix of long-run arrival probabilities in the graph
        :type probs: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling morphisms
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: A morphism from `src` to `dest`
        :rtype: :class:`discopy.biclosed.Diagram`
        """
        assert dest != Ty()

        location = src
        path = Id(src)
        dest_index = self._index(dest)
        with pyro.markov():
            while location != dest:
                generators = self._object_generators(location, True)
                if len(path) + 1 < min_depth:
                    generators = [(g, cod) for (g, cod) in generators
                                  if cod != dest]
                gens = [self._index(g) for (g, _) in generators]

                dest_probs = probs[gens][:, dest_index]
                viables = dest_probs.nonzero(as_tuple=True)[0]
                selection_probs = F.softmax(
                    dest_probs[viables].log() / (temperature + 1e-10),
                    dim=-1
                )
                generators_categorical = dist.Categorical(selection_probs)
                g_idx = pyro.sample('path_step_{%s -> %s}' % (location, dest),
                                    generators_categorical.to_event(0),
                                    infer=infer)

                gen, cod = generators[viables[g_idx.item()]]
                if isinstance(gen, callable.CallableBox):
                    morphism = gen
                else:
                    morphism = gen(probs, temperature,
                                   min_depth - len(path) - 1, infer)
                path = path >> morphism
                location = cod

        return path

    def sample_morphism(self, obj, probs, temperature, min_depth=2, infer={}):
        """Sample a morphism from the terminal object into a specified object

        :param obj: Target object
        :type obj: :class:`discopy.biclosed.Ty`
        :param probs: Matrix of long-run arrival probabilities in the graph
        :type probs: :class:`torch.Tensor`
        :param temperature: Temperature (scale parameter) for sampling morphisms
        :type temperature: :class:`torch.Tensor`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :returns: A morphism from Ty() to `obj`
        :rtype: :class:`discopy.biclosed.Diagram`
        """
        with name_count():
            if obj in self._graph.nodes:
                return self.path_between(Ty(), obj, probs, temperature,
                                         min_depth, infer)
            entries = unification.unfold_arrow(obj)
            src, dest = unification.fold_product(entries[:-1]), entries[-1]
            return self.path_between(src, dest, probs, temperature, min_depth,
                                     infer)

    def forward(self, obj, min_depth=2, infer={}, temperature=None,
                arrow_weights=None):
        """Sample a morphism from the terminal object into a specified object

        :param obj: Target object
        :type obj: :class:`discopy.biclosed.Ty`

        :param int min_depth: Minimum depth of sequential composition
        :param dict infer: Inference parameters for `pyro.sample()`

        :param temperature: Temperature (scale parameter) for sampling morphisms
        :type temperature: :class:`torch.Tensor`

        :param arrow_weights: Proposed arrow weights to combine with the
                              long-run arrival probabilities
        :type arrow_weights: :class:`torch.Tensor`

        :returns: A morphism from Ty() to `obj`
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
                dist.Gamma(self.arrow_weight_alphas,
                           self.arrow_weight_betas).to_event(1)
            )

        weights = self.diffusion_counts + self.weights_matrix(arrow_weights)
        return self.sample_morphism(obj, weights, temperature, min_depth, infer)

    def skeleton(self, skip_edges=[]):
        """Construct the skeleton graph for the underlying free category

        :param list skip_edges: List of arrows to skip

        :returns: Skeleton graph
        :rtype: :class:`nx.Digraph`
        """
        arrow_weights = dist.Beta(self.arrow_weight_alphas,
                                  self.arrow_weight_betas).mean

        skeleton = nx.DiGraph()

        for node in self._graph.nodes:
            props = {}
            if 'arrow_index' in self._graph.nodes[node]:
                k = self._graph.nodes[node]['arrow_index']
                props['weight'] = arrow_weights[k].item()
            skeleton.add_node(str(node), **props)

        for u, v in self._graph.edges:
            if (u, v) in skip_edges:
                continue
            skeleton.add_edge(str(u), str(v))
        return skeleton

    def draw(self, skip_edges=[], filename=None, notebook=False):
        """Draw the free category's nerve/skeleton using either PyVis or
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
