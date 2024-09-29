import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from net.model.encoder import PointEncoder
from net.model.attention import CrossAttention, SelfAttention, ConditionalAttention


def _init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv1d):
        trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Reconstructor(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # === TOKENIZE ===

        # point encoder (to latent space)
        self.encoder = PointEncoder(config.model.encoder)
        # aggregation of per-point features to a smaller set of tokens
        self.z_phi = nn.Embedding(self.config.dimensions.num_query_in, self.config.dimensions.latent_dim)
        self.aggregate = CrossAttention(config.model.aggregate)

        # === PROCESS ENCODE ===

        # complete/process via self attention
        self.completer = SelfAttention(config.model.process)

        # === PROCESS DECODE ===

        # interpolation of tokens at queries positions via cross attention
        self.interpolate = CrossAttention(config.model.interpolate)
        # part masks
        self.z_pi = nn.Embedding(self.config.dimensions.num_parts, self.config.dimensions.latent_dim)
        self.segmenter = CrossAttention(config.model.segment)
        emb_part_type = config.model.emb_part.type
        if emb_part_type == 'none':
            self.process_part = nn.Identity()
        elif emb_part_type == 'fc':
            self.process_part = nn.Linear(self.config.dimensions.latent_dim, self.config.dimensions.latent_dim)
        elif emb_part_type == 'mlp':
            self.process_part = nn.Sequential(
                nn.Linear(self.config.dimensions.latent_dim, self.config.dimensions.latent_dim),
                nn.ReLU(),
                nn.Linear(self.config.dimensions.latent_dim, self.config.dimensions.latent_dim),
                nn.ReLU(),
                nn.Linear(self.config.dimensions.latent_dim, self.config.dimensions.latent_dim),
            )
        else:
            raise ValueError(f'unknown emb_part {emb_part_type}')

        # === TOPOLOGY ===    
        # genus: part latent -MLP-> classification
        self.topo_g = nn.Sequential(
                nn.Linear(self.config.dimensions.latent_dim, self.config.dimensions.latent_dim//4),
                nn.ReLU(),
                nn.Linear(self.config.dimensions.latent_dim//4, self.config.dimensions.latent_dim//16),
                nn.ReLU(),
                nn.Linear(self.config.dimensions.latent_dim//16, self.config.dimensions.num_genus+1),  # +1 for outlier
        )

        # -- init weights --
        self.apply(_init_weights)

    def forward(self, observed, query, z=None, decode=True):

        if z is None:
            # == ENCODE ==
            z = self.encode(observed)  # -> B x G x C_latent

            # == COMPLETION ==
            z = self.completer(z)  # -> B x G x C_latent

        if decode:
            # == DECODE ==
            rec, g = self.decode(query, z)
        else:
            rec, g = None, None

        return rec, g, z

    def encode(self, observed):
        """
        From N observed points in Euclidean space (with label) to G tokens in latent space.

        Returns:
            z: {G} tokens of {C_latent} dimension
        """

        # from Euclidean space to latent space
        z_obs = self.encoder(observed)  # -> B x N x C_latent

        # get query latents (same for all observations)
        z_q = self.z_phi.weight.unsqueeze(0).expand(z_obs.shape[0], -1, -1)
        # aggregate to a smaller number of tokens
        z_obs_agg = self.aggregate(z_q, context=z_obs)  # -> B x G x C_latent

        # optional: add a "high-level token" as in https://github.com/NeuralCarver/michelangelo
        if self.config.model.high_level_token == 'mean':
            z_obs_agg = torch.cat([z_obs_agg, z_obs_agg.mean(dim=1, keepdim=True)], dim=1)  # -> B x (G+1) x C_latent
        elif self.config.model.high_level_token == 'max':
            z_obs_agg = torch.cat([z_obs_agg, z_obs_agg.max(dim=1, keepdim=True)[0]], dim=1)  # -> B x (G+1) x C_latent

        return z_obs_agg
    
    def decode_topology(self, z):
        # ~~~ GLOBAL ~~~
        # == SEGMENT: "global" part tokens ==
        z_p = self.segmenter(self.z_pi.weight.unsqueeze(0).expand(z.shape[0], -1, -1), context=z)  # -> B x max parts x C_latent
        z_p = self.process_part(z_p)  # -> B x max parts x C_latent

        # == TOPOLOGY: predict component and genus ==
        g_logit = self.topo_g(z_p)

        return g_logit

    def decode(self, query, z):
        # ~~~ GLOBAL ~~~
        # == SEGMENT: "global" part tokens ==
        z_p = self.segmenter(self.z_pi.weight.unsqueeze(0).expand(z.shape[0], -1, -1), context=z)  # -> B x max parts x C_latent
        z_p = self.process_part(z_p)  # -> B x max parts x C_latent

        # == TOPOLOGY: predict component and genus ==
        g_logit = self.topo_g(z_p)
        
        # ~~~ LOCAL ~~~
        # == INTERPOLATE: get tokens at query locations ("local") ==
        z_q = self.encoder(query)
        # get interpolated tokens at query points
        z_q_int = self.interpolate(z_q, context=z, decode=True)  # -> B x query x C_latent

        # == PREDICT: decode latent to part logits (incl "outlier" part for non-occupied points) ==
        # dot global part tokens with local sample tokens = part logit per sample
        part_logit = torch.einsum("bcp,bqc->bqp", z_p.transpose(2, 1), z_q_int)  # -> B x query x max parts

        return part_logit, g_logit
    

class Predictor(Reconstructor):

    def __init__(self, config) -> None:
        super().__init__(config)

        # === PREDICT ===
        # inject ee/action via alternating self and cross attention
        self.condition = ConditionalAttention(config.model.condition)

        # -- init weights --
        self.apply(_init_weights)

    def reconstruct(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, ee, query, z_cur, decode=False):

        # == ACTION ==
        # get ee shape's latent representation
        if len(ee.shape) == 3:
            z_e = self.encode(ee).detach()  # -> B x G x C_latent
        else:  # observed and target, concatenated along points along extra dim
            z_e_observed = self.encode(ee[0]).detach()  # -> B x G x C_latent
            z_e_target = self.encode(ee[1]).detach()  # -> B x G x C_latent
            z_e = torch.cat([z_e_observed, z_e_target], dim=1)  # -> B x (2G) x C_latent

        # == PREDICT NEXT ==
        # apply condition
        z_nxt = self.condition(z_cur, context=z_e)  # -> B x G x C_latent

        if decode:
            # == DECODE NEXT ==
            pred, g = self.decode(query, z_nxt)
        else:
            pred, g = None, None

        return pred, g, z_nxt
