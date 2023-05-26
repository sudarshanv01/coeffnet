from monty.serialization import loadfn, dumpfn

from instance_mongodb import instance_mongodb_sei


if __name__ == "__main__":
    """Write out a yaml file with the angle information."""

    db = instance_mongodb_sei(project="mlts")
    collection = db["rotated_sn2_reaction_calculation"]

    data_to_store = {}

    for doc in collection.find({"tags.state": "transition_state"}):
        idx = doc["tags"]["idx"]
        euler_angles = doc["tags"]["euler_angles"]

        data_to_store[idx] = euler_angles

    dumpfn(data_to_store, "input/idx_to_euler_angles.yaml")
