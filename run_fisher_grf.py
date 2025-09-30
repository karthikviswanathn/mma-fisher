from fisher_simulator import FisherSimulator

simulator = FisherSimulator(n_cov=5000, n_der=5000, load_path = 'data/grf_5k.npz', save_path = 'data/grf_5k.npz', val_split=0.)

history = simulator.train_model(epochs = 1000, lr=1e-3)