# Input format

There is a specific input format that is required for proper training and inference. In this section, we list the expected format for each supported electronic structure code. 

## Q-Chem

### Setting up

If you intend to run DFT calculations with QChem, we highly recommend that you use the associated [pymatgen](https://github.com/materialsproject/pymatgen) / [atomate](https://github.com/hackingmaterials/atomate) / [fireworks](https://github.com/materialsproject/fireworks) codes. Most input / output of the matrices required to run the model are already implemented within these codes.

```{admonition} Note
Througout this section we will assume that you are using [pymatgen](https://github.com/materialsproject/pymatgen) / [atomate](https://github.com/hackingmaterials/atomate) / [fireworks](https://github.com/materialsproject/fireworks) codes to perform the required calculations.
```

There are two manipulations required to make the ordering of axes consistent between Q-Chem and this code. 

1. Switch the $x, y, z$ axis of the molecule with $z, x, y$. To make your `Molecule` object comparable with this transformation, starting from a `molecule`,  
```python
coordinates = np.array(molecule.cart_coords)
coordinates[:, [0, 1, 2]] = coordinates[:, [2, 0, 1]]

molecule = Molecule(
    species=molecule.species,
    coords=coordinates,
    charge=molecule.charge,
    spin_multiplicity=molecule.spin_multiplicity,
)
```
2. Add `purecart=1111` to the `rem` of your Q-Chem input file. Further information may be found within the official [Q-Chem documentation](https://manual.q-chem.com/4.3/sect-userbasis.html).


### Input to model
Once your calculation has completed, you may directly convert your MongoDB taskdocument to a JSON file which can be directly fed into the model using `TaskdocsToData`. 

```python
taskdocs_to_data = TaskdocsToData(
    collection=collection,
    filter_collection={'tags.inverted_coordinates': True},
    identifier="idx",
    state_identifier="state",
    reactant_tag="reactant",
    transition_state_tag="transition_state",
    product_tag="product",
    basis_set_type="full",
    basis_info_raw=basis_info,
    d_functions_are_spherical=True,
    **kwargs
)
```

- `collection` is the collection in which your taskdocuments are stored.
- `filter_collection` is the dictionary that will be passed to `collection.find(**filter_collection)`. This option is useful if there are many calculations in the same collection and you would like to pick just a handful. 
- `identifier` is a unique identifier for a particular reaction.
- `state_identifier` is a unique identifier for the state (reactant, product or transition state).
- `reactant_tag` is the tag assigned for reactant calculations
- `product_tag` is the tag assigned for product calculations
- `transition_state_tag` is the tag assined for transition state calculations
- `basis_set_type` is the type of basis set you would like to use, either `minimal` or `full`
- `basis_info_raw` is a dictionary from a JSON file-format coming from [`basissetexchange`](https://www.basissetexchange.org/).
- `d_functions_are_spherical`: Decide if _d_ functions are spherical or cartesian

```{admonition} Note
Currently only $l\leq2$, i.e. up to _d_-orbitals are supported. Only spherical versions of these orbitals are currently implemented.
```