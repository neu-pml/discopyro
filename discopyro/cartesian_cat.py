import collections
from discopy import Ty
import functools
from indexed import IndexedOrderedDict
import networkx as nx
import pyro
from pyro.contrib.autoname import name_count
import pyro.distributions as dist
import pyro.nn as pnn
import pytorch_expm.expm_taylor as expm
import torch
import torch.distributions.constraints as constraints
import torch.nn as nn
import torch.nn.functional as F

from . import closed

NONE_DEFAULT = collections.defaultdict(lambda: None)

class CartesianCategory(pyro.nn.PyroModule):
    def __init__(self, generators, global_elements):
        super().__init__()
        self._graph = nx.DiGraph()
        for i, gen in enumerate(generators):
            assert isinstance(gen, closed.TypedFunction)

            if gen.typed_dom not in self._graph:
                self._graph.add_node(gen.typed_dom, index=len(self._graph))
            self._graph.add_node(gen, index=len(self._graph), arrow_index=i)
            if gen.typed_cod not in self._graph:
                self._graph.add_node(gen.typed_cod, index=len(self._graph))
            self._graph.add_edge(gen.typed_dom, gen)
            self._graph.add_edge(gen, gen.typed_cod)

            if isinstance(gen.function, nn.Module):
                self.add_module('generator_%d' % i, gen.function)
            if isinstance(gen, closed.TypedDaggerFunction):
                dagger = gen.dagger()
                if isinstance(dagger.function, nn.Module):
                    self.add_module('generator_%d_dagger' % i, dagger.function)

        for i, obj in enumerate(self.obs):
            self._graph.nodes[obj]['object_index'] = i
            self._graph.nodes[obj]['global_elements'] = []

        for elem in global_elements:
            assert isinstance(elem, closed.TypedFunction)
            assert elem.typed_dom == closed.TOP
            elem = closed.TypedDaggerFunction(elem.typed_dom, elem.typed_cod,
                                              elem.function, lambda *args: None)

            if isinstance(elem.function, nn.Module):
                i = self._graph.nodes[elem.typed_cod]['object_index']
                k = len(self._graph.nodes[elem.typed_cod]['global_elements'])
                self.add_module('global_element_%d_%d' % (i, k), elem.function)
            self._graph.nodes[elem.typed_cod]['global_elements'].append(elem)

        max_elements = max([len(self._graph.nodes[obj]['global_elements'])
                            for obj in self.obs])
        self.global_element_weights = pnn.PyroParam(
            torch.ones(len(self.obs), max_elements),
            constraint=constraints.positive
        )
        self.arrow_distances = pnn.PyroParam(torch.ones(len(self.ars)),
                                             constraint=constraints.positive)
        self.confidence_alpha = pnn.PyroParam(torch.ones(1),
                                              constraint=constraints.positive)
        self.confidence_beta = pnn.PyroParam(torch.ones(1),
                                             constraint=constraints.positive)

    @property
    def param_shapes(self):
        return (self.global_element_weights.shape, self.arrow_distances.shape,
                self.confidence_alpha.shape * 2)

    def _object_generators(self, obj, forward=True, arrow_distances=None):
        if arrow_distances is None:
            arrow_distances = self.arrow_distances
        edges = self._graph.out_edges if forward else self._graph.in_edges
        dir_index = 1 if forward else 0
        generators = []
        arrow_indices = []
        for edge in edges(obj):
            gen = edge[dir_index]
            generators.append(gen)
            arrow_indices.append(self._graph.nodes[gen]['arrow_index'])
        return generators, arrow_distances[arrow_indices]

    def _object_elements(self, obj, global_element_weights=None):
        if global_element_weights is None:
            global_element_weights = self.global_element_weights
        index = self._graph.nodes[obj]['object_index']
        elements = self._graph.nodes[obj]['global_elements']
        weights = global_element_weights[index, :len(elements)]
        return elements, weights

    @property
    def obs(self):
        return [node for node in self._graph
                if isinstance(node, closed.CartesianClosed)]

    @property
    def ars(self):
        return [node for node in self._graph
                if isinstance(node, closed.TypedFunction)]

    @pnn.pyro_method
    def diffusion_distances(self, arrow_distances=None):
        if arrow_distances is None:
            arrow_distances = self.arrow_distances
        transitions = arrow_distances.new_zeros([len(self._graph)] * 2)

        row_indices = []
        column_indices = []
        distances = []
        for arrow in self.ars:
            dom, cod = arrow.typed_dom, arrow.typed_cod

            row_indices.append(self._graph.nodes[dom]['index'])
            column_indices.append(self._graph.nodes[arrow]['index'])
            k = self._graph.nodes[arrow]['arrow_index']
            distances.append(-arrow_distances[k])

            row_indices.append(self._graph.nodes[arrow]['index'])
            column_indices.append(self._graph.nodes[cod]['index'])
            distances.append(-arrow_distances.new_ones(1))

        transitions = transitions.index_put((torch.LongTensor(row_indices),
                                             torch.LongTensor(column_indices)),
                                            torch.stack(distances, dim=0).exp())
        transitions = transitions / transitions.sum(dim=1, keepdim=True)
        diffusions = expm.expm(transitions.unsqueeze(0)).squeeze(0)
        return -torch.log(diffusions)

    @pnn.pyro_method
    def product_arrow(self, ty, depth=0, min_depth=0, infer={},
                      confidence=None, params=NONE_DEFAULT):
        entries = [self.forward(closed.wrap_base_ob(obj), depth + 1, min_depth,
                                infer, confidence, params)
                   for obj in ty.objects]
        return functools.reduce(lambda f, g: f.tensor(g), entries)

    @pnn.pyro_method
    def path_between(self, src, dest, confidence, min_depth=0, infer={},
                     params=NONE_DEFAULT):
        assert src != closed.TOP
        assert dest != closed.TOP

        location = src
        distances = self.diffusion_distances(params['arrow_distances'])

        path = []
        with pyro.markov():
            while location != dest and len(path) < min_depth:
                generators, _ = self._object_generators(
                    location, True, params['arrow_distances']
                )
                gen_indices = [(self._graph.nodes[g]['index'],
                                self._graph.nodes[dest]['index'])
                               for g in generators]
                distances_to_dest = distances[gen_indices]
                generators_categorical = dist.Categorical(
                    probs=F.softmin(confidence * distances_to_dest, dim=0)
                ).to_event(0)
                g_idx = pyro.sample('path_step_{%s -> %s}' % (location, dest),
                                    generators_categorical, infer=infer)
                path.append(generators[g_idx.item()])

        return functools.reduce(lambda f, g: f >> g, path)

    def forward(self, obj, depth=0, min_depth=0, infer={}, confidence=None,
                params=NONE_DEFAULT):
        with name_count():
            if confidence is None:
                confidence = pyro.sample(
                    'distances_confidence',
                    dist.Gamma(self.confidence_alpha,
                               self.confidence_beta).to_event(0)
                )

            generators, distances = self._object_generators(
                obj, False, params['arrow_distances']
            )
            if obj.is_compound():
                generators.append(None)
                distances = torch.cat((distances, distances.new_ones(1)), dim=0)
            if depth >= min_depth:
                elements, weights = self._object_elements(
                    obj, params['global_element_weights']
                )
                generators = generators + elements
                distances = torch.cat((distances + depth, -weights),
                                      dim=0)

            generators_cat = dist.Categorical(
                probs=F.softmin(distances * confidence, dim=0)
            )
            g_idx = pyro.sample('generator_{-> %s}' % obj, generators_cat,
                                infer=infer)
            generator = generators[g_idx.item()]

            if generator is None:
                result = obj.match(
                    base=lambda ty: self.product_arrow(ty, depth, min_depth,
                                                       infer, confidence,
                                                       params),
                    var=lambda v: closed.UnificationException(None, None, v),
                    arrow=lambda l, r: self.path_between(l, r, confidence,
                                                         min_depth=min_depth,
                                                         params=params)
                )
            elif generator.typed_dom == closed.TOP and depth >= min_depth:
                result = generator
            else:
                predecessor = self.forward(generator.typed_dom, depth + 1,
                                           min_depth, infer,
                                           confidence=confidence, params=params)
                result = predecessor >> generator

            return result

    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])