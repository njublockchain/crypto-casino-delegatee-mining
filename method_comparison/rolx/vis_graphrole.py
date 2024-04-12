# %%
import os 
import pickle
import networkx as nx
from graphrole import RecursiveFeatureExtractor, RoleExtractor

# address = "0xc2a81eb482cb4677136d8812cc6db6e0cb580883"
# role_extractor: RoleExtractor = pickle.load(open(f"roles_workspace/{address}.pkl", "rb"))

filetype = "gexf"

filenames = os.listdir("./roles_workspace")
for filename in filenames:
    address = filename.split(".")[0]
    if os.path.exists(f"./roles_fig/{address}.{filetype}"):
        continue
    role_extractor: RoleExtractor = pickle.load(open(f"roles_workspace/{address}.pkl", "rb"))
    nxG: nx.MultiDiGraph = pickle.load(
        open(
            f"./pos_nx_normalized/{address}.pkl",
            "rb",
        )
    )

    roles = role_extractor.roles
    print(roles)
    # %%
    roles = { k: int(v.removeprefix("role_")) for k,v in roles.items()}

    # %%
    nx.set_node_attributes(
        nxG,
        roles,
        name="roles"
    )

    # %%
    for role_name in role_extractor.role_percentage:
        pers = role_extractor.role_percentage.loc[:, role_name].to_dict()
        nx.set_node_attributes(nxG, pers, name=role_name)

    # nx.write_gexf(nxG, f"./roles_fig/{address}.gexf", prettyprint=True)
    nx.write_gexf(nxG, f"./roles_fig/{address}.{filetype}")
