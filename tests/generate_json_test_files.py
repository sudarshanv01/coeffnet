from collections import defaultdict

import json

from monty.serialization import loadfn, dumpfn

import basis_set_exchange as bse

from instance_mongodb import instance_mongodb_sei

if __name__ == "__main__":

    db = instance_mongodb_sei(project="mlts")
    collection = db.rotated_water_calculations
    basis_sets = collection.distinct("orig.rem.basis")
    data = defaultdict(lambda: defaultdict(list))

    for basis_set in basis_sets:
        basis_info = bse.get_basis(basis_set, fmt="json", elements=[1, 8])
        basis_info = json.loads(basis_info)

    electron_shells = {
        k: basis_info["elements"][k]["electron_shells"]
        for k in basis_info["elements"].keys()
    }
    angular_momenta = {
        k: [shell["angular_momentum"] for shell in electron_shells[k]]
        for k in electron_shells.keys()
    }
    angular_momenta = {
        k: [item for sublist in angular_momenta[k] for item in sublist]
        for k in angular_momenta.keys()
    }

    find_tags = {"orig.rem.basis": basis_set}

    for doc in collection.find(find_tags).sort("tags.idx", 1):
        _alpha_coeff_matrix = doc["calcs_reversed"][0]["alpha_coeff_matrix"]
        _alpha_eigenvalues = doc["calcs_reversed"][0]["alpha_eigenvalues"]
        _alpha_fock_matrix = doc["calcs_reversed"][0]["alpha_fock_matrix"]

        base_quantities_qchem = BaseQuantitiesQChem(
            fock_matrix=_alpha_fock_matrix,
            eigenvalues=_alpha_eigenvalues,
            coeff_matrix=_alpha_coeff_matrix,
        )

        _alpha_ortho_coeff_matrix = base_quantities_qchem.get_ortho_coeff_matrix()
        molecule = Molecule.from_dict(doc["orig"]["molecule"])
        symbols = [site.specie.symbol for site in molecule.sites]
        atom_numbers = [atomic_numbers[symbol] for symbol in symbols]
        _basis_functions_orbital = []

        _irreps = ""
        for atom_number in atom_numbers:
            _angular_momenta = angular_momenta[str(atom_number)]
            _angular_momenta = np.array(_angular_momenta)
            _basis_functions = 2 * _angular_momenta + 1
            for _basis_function in _basis_functions:
                if _basis_function == 1:
                    _basis_functions_orbital.extend(["s"])
                    _irreps += "+1x0e"
                elif _basis_function == 3:
                    _basis_functions_orbital.extend(["p", "p", "p"])
                    _irreps += "+1x1o"
                elif _basis_function == 5 and basis_set != "6-31g*":
                    _basis_functions_orbital.extend(["d", "d", "d", "d", "d"])
                    _irreps += "+1x2e"
                elif _basis_function == 5 and basis_set == "6-31g*":
                    _basis_functions_orbital.extend(["s", "d", "d", "d", "d", "d"])
                    _irreps += "+1x0e+1x2e"

        _irreps = _irreps[1:]

        data[basis_set]["alpha_coeff_matrix"].append(_alpha_coeff_matrix)
        data[basis_set]["alpha_eigenvalues"].append(_alpha_eigenvalues)
        data[basis_set]["alpha_fock_matrix"].append(_alpha_fock_matrix)
        data[basis_set]["alpha_ortho_coeff_matrix"].append(_alpha_ortho_coeff_matrix)
        data[basis_set]["basis_functions_orbital"].append(_basis_functions_orbital)
        data[basis_set]["alpha_overlap_matrix"].append(
            base_quantities_qchem.overlap_matrix
        )
        data[basis_set]["irreps"].append(_irreps)
        data[basis_set]["euler_angles"].append(doc["tags"]["euler_angles"])
        data[basis_set]["idx"].append(doc["tags"]["idx"])
        data[basis_set]["molecule"].append(molecule)
