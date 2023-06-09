import numpy.typing as npt

import plotly.express as px
import plotly.graph_objects as go

from ase.data import atomic_numbers, atomic_names, atomic_masses, vdw_radii
from ase.data.colors import jmol_colors

from minimal_basis.postprocessing.transformations import (
    OrthoCoeffMatrixToGridQuantities,
    NodeFeaturesToOrthoCoeffMatrix,
    DatapointStoredVectorToOrthogonlizationMatrix,
)


def get_instance_grid(
    data,
    node_features,
    basis_name: str = "6-31g*",
    charge: int = -1,
    grid_points_per_axis: int = 10,
    buffer_grid: int = 5,
    uses_cartesian_orbitals: bool = True,
):

    datapoint_to_orthogonalization_matrix = (
        DatapointStoredVectorToOrthogonlizationMatrix(
            data.orthogonalization_matrix_transition_state
        )
    )
    datapoint_to_orthogonalization_matrix()

    orthogonalization_matrix_transition_state = (
        datapoint_to_orthogonalization_matrix.get_orthogonalization_matrix()
    )
    nodefeatures_to_orthocoeffmatrix = NodeFeaturesToOrthoCoeffMatrix(
        node_features=node_features,
        mask=data.basis_mask,
    )
    nodefeatures_to_orthocoeffmatrix()

    ortho_coeff_matrix = nodefeatures_to_orthocoeffmatrix.get_ortho_coeff_matrix()
    ortho_coeff_matrix_to_grid_quantities = OrthoCoeffMatrixToGridQuantities(
        ortho_coeff_matrix=ortho_coeff_matrix,
        orthogonalization_matrix=orthogonalization_matrix_transition_state,
        positions=data.pos_transition_state,
        species=data.species,
        basis_name=basis_name,
        indices_to_keep=data.indices_to_keep,
        charge=charge,
        uses_carterian_orbitals=uses_cartesian_orbitals,
        buffer_grid=buffer_grid,
        grid_points=grid_points_per_axis,
    )
    ortho_coeff_matrix_to_grid_quantities()

    return ortho_coeff_matrix_to_grid_quantities


def add_grid_to_fig(
    grid: npt.ArrayLike,
    molecular_orbital,
    fig: go.Figure,
    isomin: float,
    isomax: float,
    cmap: str = "cividis",
    surface_count: int = 3,
    species: npt.ArrayLike = None,
    positions: npt.ArrayLike = None,
):

    fig_go = go.Figure(
        data=go.Isosurface(
            x=grid[:, 0],
            y=grid[:, 1],
            z=grid[:, 2],
            value=molecular_orbital.flatten(),
            isomin=isomin,
            isomax=isomax,
            surface_count=surface_count,
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.5,
            colorbar=dict(
                title="Molecular orbital",
                titleside="right",
                titlefont=dict(size=18),
                tickfont=dict(size=14),
            ),
            colorscale=cmap,
        )
    )
    for figdata in fig_go.data:
        fig.add_trace(figdata)

    colors_of_atom = [
        jmol_colors[int(species)]
        for species in species.view(-1).detach().numpy().flatten()
    ]
    colors_of_atom = [
        f"rgb({color[0]*255}, {color[1]*255}, {color[2]*255})"
        for color in colors_of_atom
    ]

    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                color=colors_of_atom,
            ),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            xaxis_title="",
            yaxis_title="",
            zaxis_title="",
        ),
        template="plotly_dark",
    )
