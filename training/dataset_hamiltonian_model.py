import os
import logging

import torch_geometric
import networkx as nx

from minimal_basis.dataset.dataset_hamiltonian import HamiltonianDataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """Test the DataPoint class."""
    JSON_FILE = "input_files/output_QMrxn20_debug.json"
    BASIS_FILE = "input_files/sto-3g.json"
    GRAPH_GENERTION_METHOD = "sn2"

    logging.basicConfig(filename="dataset.log", filemode="w", level=logging.DEBUG)

    data_point = HamiltonianDataset(
        JSON_FILE, BASIS_FILE, graph_generation_method=GRAPH_GENERTION_METHOD
    )
    data_point.load_data()
    data_point.parse_basis_data()
    datapoint = data_point.get_data()

    # Graph the dataset
    for i, data in enumerate(datapoint):
        # Plot the graph for each datapoint
        graph = torch_geometric.utils.to_networkx(data)
        # Iteratively add subplots to a figure
        fig = plt.figure()
        for j, nx_graph in enumerate(nx.weakly_connected_components(graph)):
            # Get the number of reactants and products based on the len of x
            n_react_prod = len(data.x)
            # Split n_react_prod into num_x and num_y
            # to make sure the plots look nice
            num_x = int(np.ceil(np.sqrt(n_react_prod)))
            num_y = int(np.ceil(n_react_prod / num_x))
            # Add subplot to figure
            ax = fig.add_subplot(num_x, num_y, j + 1)
            nx.draw(
                graph.subgraph(nx_graph),
                pos=nx.spring_layout(graph.subgraph(nx_graph), seed=42),
                with_labels=False,
                cmap="Set2",
                ax=ax,
            )

        fig.savefig(os.path.join("output", f"graph_{i}.png"), dpi=300)
        plt.close(fig)

        # Also plot the Hamiltonian for each spin for each spin for each
        # molecules in a separate subplot. Ensure that each node feature
        # is highlighted (in a bounding axv/axhline).
        fig, ax = plt.subplots(
            2,
            len(data["x"]),
            figsize=(4 * len(data["x"]), 4),
            squeeze=False,
            constrained_layout=True,
        )
        # Plot the coupling elements, i.e. all the elements are not on the diagonal
        figd, axd = plt.subplots(
            2,
            len(data["edge_attr"]),
            figsize=(4 * len(data["edge_attr"]), 4),
            squeeze=False,
            constrained_layout=True,
        )

        for j, state_index in enumerate(data["x"]):

            # Each state_index gets its own subplot
            H_up = data["x"][state_index][..., 0]
            H_down = data["x"][state_index][..., 1]
            H_up = H_up.numpy()
            H_down = H_down.numpy()

            V_up = data["edge_attr"][state_index][..., 0]
            V_down = data["edge_attr"][state_index][..., 1]
            V_up = V_up.numpy()
            V_down = V_down.numpy()

            # Plot the Hamiltonian for each spin separately
            cax1 = ax[0, j].imshow(
                H_up, cmap="viridis", interpolation="none", vmin=-5, vmax=5
            )
            cax2 = ax[1, j].imshow(
                H_down, cmap="viridis", interpolation="none", vmin=-5, vmax=5
            )
            # Add a colorbar to each subplot
            fig.colorbar(cax1, ax=ax[0, j])
            fig.colorbar(cax2, ax=ax[1, j])

            # Plot the coupling elements, i.e. all the elements are not on the diagonal
            cax1 = axd[0, j].imshow(
                V_up, cmap="viridis", interpolation="none", vmin=-5, vmax=5
            )
            cax2 = axd[1, j].imshow(
                V_down, cmap="viridis", interpolation="none", vmin=-5, vmax=5
            )
            # Add a colorbar to each subplot
            figd.colorbar(cax1, ax=axd[0, j])
            figd.colorbar(cax2, ax=axd[1, j])

            # Highlight node specific features in the matrix
            # print(data['atom_basis_functions'][state_index])
            for (start, stop) in data["indices_atom_basis"][state_index]:
                # Highlight the starting index
                ax[0, j].axvline(start, color="k")
                ax[1, j].axvline(start, color="k")
                ax[0, j].axhline(start, color="k")
                ax[1, j].axhline(start, color="k")
                ax[0, j].set_title(f"State {state_index}, spin up")
                ax[1, j].set_title(f"State {state_index}, spin down")
                axd[0, j].axvline(start, color="k")
                axd[1, j].axvline(start, color="k")
                axd[0, j].axhline(start, color="k")
                axd[1, j].axhline(start, color="k")
                axd[0, j].set_title(f"State {state_index}, spin up")
                axd[1, j].set_title(f"State {state_index}, spin down")

        for a in ax.flat:
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect("equal")

        for a in axd.flat:
            a.set_xticks([])
            a.set_yticks([])
            a.set_aspect("equal")

        fig.savefig(os.path.join("output", f"hamiltonian_graph_{i}.png"), dpi=300)
        figd.savefig(os.path.join("output", f"coupling_graph_{i}.png"), dpi=300)
        plt.close(fig)
