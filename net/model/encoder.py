import torch
import torch.nn as nn


class PointEncoder(nn.Module):

    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config

        embedding_dim = config.embedding_dim
        assert embedding_dim % 6 == 0
        embedding_half_size = embedding_dim // 2
        basis_dim = embedding_dim // 6

        # --- input encoding
        if config.type == 'rbf':
            # adapted from https://github.com/1zb/3DShape2VecSet
            bvals = torch.pow(2, torch.arange(basis_dim)).float() * torch.pi
            bvals = torch.stack([
                torch.cat([bvals, torch.zeros(basis_dim), torch.zeros(basis_dim)]),
                torch.cat([torch.zeros(basis_dim), bvals, torch.zeros(basis_dim)]),
                torch.cat([torch.zeros(basis_dim), torch.zeros(basis_dim), bvals]),
            ])
        else:
            raise ValueError(f'unknown embedding_method {config.embedding_method}')
        self.register_buffer('basis', bvals)

        # --- feature embedding
        net = nn.Linear
        encoding_dim = embedding_dim + 3  # we always append the original xyz to the encoding
        # with features (xyz + onehot label; observation will use this mlp if feature_dim > 0)
        self.mlp = net(encoding_dim, config.output_dim - config.feature_dim)  # if there are features, we append them after mlp
        # just positions (ee and query will use this mlp; if feature_dim == 0, also the observation)
        self.mlp_xyz = net(encoding_dim, config.output_dim) 

    def forward(self, observed):
        if observed.shape[-1] == 4:
            assert self.config.feature_dim > 0
            xyz = observed[..., :3]
            labels = observed[..., 3]
            features = torch.nn.functional.one_hot(labels.long(),
                                                   num_classes=self.config.feature_dim).float()  # dim = num parts
        else:
            xyz = observed
            features = None
        
        xyz_features = torch.cat([self.encode(xyz), xyz], dim=-1)
        if observed.shape[-1] == 3:
            xyz_features = self.mlp_xyz(xyz_features)
        else:
            xyz_features = torch.cat([self.mlp(xyz_features), features], dim=-1)
        return xyz_features
    
    def encode(self, xyz):
        projections = torch.einsum('bnd,de->bne', xyz, self.basis)
        return torch.cat([projections.sin(), projections.cos()], dim=-1)
