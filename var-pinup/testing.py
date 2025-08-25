from config import *

#cfg1 = ExperimentConfig(sym_links=[(0,1), (2,3)], directed=False)

#print(build_gains(cfg1))

cfg2 = ExperimentConfig(Sigma=0.02, 
                        self_memory=0.35, 
                        link_gains={(0,1):0.9, (1,2):0.5}, 
                        directed=False
                        )

print(build_A0(cfg2))