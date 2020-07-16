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
        self._graph.add_node(closed.TOP, index=0, object_index=0)
        for i, gen in enumerate(generators):
            assert isinstance(gen, closed.TypedBox)

            if gen.typed_dom not in self._graph:
                self._graph.add_node(gen.typed_dom, index=len(self._graph))
            self._graph.add_node(gen, index=len(self._graph), arrow_index=i)
            if gen.typed_cod not in self._graph:
                self._graph.add_node(gen.typed_cod, index=len(self._graph))
            self._graph.add_edge(gen.typed_dom, gen)
            self._graph.add_edge(gen, gen.typed_cod)

            if isinstance(gen.function, nn.Module):
                self.add_module('generator_%d' % i, gen.function)
            if isinstance(gen, closed.TypedDaggerBox):
                dagger = gen.dagger()
                if isinstance(dagger.function, nn.Module):
                    self.add_module('generator_%d_dagger' % i, dagger.function)

        for i, obj in enumerate(self.obs):
            self._graph.nodes[obj]['object_index'] = i + 1

        for i, elem in enumerate(global_elements):
            assert isinstance(elem, closed.TypedBox)
            assert elem.typed_dom == closed.TOP
            elem = closed.TypedDaggerBox(elem.name, elem.typed_dom,
                                         elem.typed_cod, elem.function,
                                         lambda *args: None)

            self._graph.add_node(elem, index=len(self._graph),
                                 arrow_index=len(generators) + i)
            self._graph.add_edge(closed.TOP, elem)
            self._graph.add_edge(elem, elem.typed_cod)

            if isinstance(elem.function, nn.Module):
                self.add_module('global_element_%d' % i, elem.function)

        for i, obj in enumerate(self.compound_obs):
            if obj._key == closed.CartesianClosed._Key.ARROW:
                src, dest = obj.arrow()
                def macro(probs, min_depth, infer, l=src, r=dest):
                    return self.path_between(l, r, probs, min_depth, infer)
            elif obj._key == closed.CartesianClosed._Key.BASE:
                def macro(probs, min_depth, infer, obj=obj):
                    return self.product_arrow(obj, probs, min_depth, infer)

            arrow_index = len(generators) + len(global_elements) + i
            self._graph.add_node(macro, index=len(self._graph),
                                 arrow_index=arrow_index)
            self._graph.add_edge(closed.TOP, macro)
            self._graph.add_edge(macro, obj)

        self.arrow_distances = pnn.PyroParam(torch.ones(len(self.ars)),
                                             constraint=constraints.positive)
        self.confidence_alpha = pnn.PyroParam(torch.ones(1),
                                              constraint=constraints.positive)
        self.confidence_beta = pnn.PyroParam(torch.ones(1),
                                             constraint=constraints.positive)

    @property
    def param_shapes(self):
        return (self.arrow_distances.shape, self.confidence_alpha.shape * 2)

    def _object_generators(self, obj, forward=True):
        edges = self._graph.out_edges if forward else self._graph.in_edges
        dir_index = 1 if forward else 0
        generators = []
        arrow_indices = []
        for edge in edges(obj):
            gen = edge[dir_index]
            generators.append(gen)
            arrow_indices.append(self._graph.nodes[gen]['arrow_index'])
        return generators

    @property
    def obs(self):
        return [node for node in self._graph
                if isinstance(node, closed.CartesianClosed)]

    @property
    def compound_obs(self):
        for obj in self.obs:
            if obj.is_compound():
                yield obj

    @property
    def ars(self):
        return [node for node in self._graph
                if isinstance(node, closed.TypedBox)]

    @pnn.pyro_method
    def diffusion_probs(self, arrow_distances):
        transitions = torch.eye(len(self._graph), device=arrow_distances.device)

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
            distances.append(arrow_distances.new_zeros(()))

        transitions = transitions.index_put((torch.LongTensor(row_indices),
                                             torch.LongTensor(column_indices)),
                                            torch.stack(distances, dim=0).exp())
        transitions = transitions / transitions.sum(dim=-1, keepdim=True)
        diffusions = expm.expm(transitions.unsqueeze(0)).squeeze(0)
        diffusions_sum = diffusions.sum(dim=-1, keepdim=True)
        return diffusions / diffusions_sum

    def diffusion_distances(self, arrow_distances):
        return -torch.log(self.diffusion_probs(arrow_distances))

    @pnn.pyro_method
    def product_arrow(self, ty, probs, min_depth=0, infer={}):
        entries = [self.sample_morphism(closed.wrap_base_ob(obj), probs,
                                        min_depth, infer)
                   for obj in ty.objects]
        return functools.reduce(lambda f, g: f.tensor(g), entries)

    @pnn.pyro_method
    def path_between(self, src, dest, probs, min_depth=0, infer={}):
        assert dest != closed.TOP

        location = src
        path = []
        with pyro.markov():
            while location != dest:
                generators = self._object_generators(location, True)
                if len(path) + 1 < min_depth:
                    generators = [g for g in generators if g.typed_cod != dest]
                gens = [self._graph.nodes[g]['index'] for g in generators]
                dest_index = self._graph.nodes[dest]['index']
                dest_probs = probs[gens][:, dest_index]
                generators_categorical = dist.Categorical(dest_probs)
                g_idx = pyro.sample('path_step_{%s -> %s}' % (location, dest),
                                    generators_categorical.to_event(0),
                                    infer=infer)
                if isinstance(generators[g_idx.item()], closed.TypedBox):
                    generator = generators[g_idx.item()]
                else:
                    macro = generators[g_idx.item()]
                    generator = macro(probs, min_depth, infer)
                path.append(generator)
                location = generator.typed_cod

        return functools.reduce(lambda f, g: f >> g, path)

    def sample_morphism(self, obj, probs, min_depth=2, infer={}):
        with name_count():
            entries = closed.unfold_arrow(obj)
            if len(entries) == 1:
                return self.path_between(closed.TOP, obj, probs, min_depth,
                                         infer)
            src, dest = closed.fold_product(entries[:-1]), entries[-1]
            return self.path_between(src, dest, probs, min_depth, infer)

    def forward(self, obj, min_depth=2, infer={}, confidence=None,
                arrow_distances=None):
        if confidence is None:
            confidence = pyro.sample(
                'distances_confidence',
                dist.Gamma(self.confidence_alpha,
                           self.confidence_beta).to_event(0)
            )
        if arrow_distances is None:
            arrow_distances = self.arrow_distances

        arrow_distances = confidence * arrow_distances
        probs = self.diffusion_probs(arrow_distances)

        return self.sample_morphism(obj, probs, min_depth, infer)


    def resume_from_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        checkpoint = torch.load(resume_path)
        self.load_state_dict(checkpoint['state_dict'])
