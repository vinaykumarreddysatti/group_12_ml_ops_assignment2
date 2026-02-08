import time
import torch
import matplotlib.pyplot as plt
import random

from src.model import LogisticModel
from src.utils import load_data, evaluate
from src.client import local_train, get_model_update
from src.server import aggregate

CLIENT_SETTINGS = [1, 2, 4, 8]
ROUNDS = 5
LOCAL_EPOCHS = 1
COMM_PROB = 0.5
COMPRESSION_RATIO = 0.1

times = []
accuracies = []
communications = []

for num_clients in CLIENT_SETTINGS:
    print(f"\nRunning with {num_clients} clients")

    client_data = load_data(num_clients)
    global_model = LogisticModel()

    start = time.time()
    total_sent = 0
    total_params = 0

    for r in range(ROUNDS):
        updates = []

        for i in range(num_clients):
            if random.random() > COMM_PROB:
                continue

            local_model = LogisticModel()
            local_model.load_state_dict(global_model.state_dict())

            local_model = local_train(local_model, client_data[i], LOCAL_EPOCHS)

            update, sent, total = get_model_update(global_model, local_model, COMPRESSION_RATIO)

            total_sent += sent
            total_params = total
            updates.append(update)

        if updates:
            global_model = aggregate(global_model, updates)

    end = time.time()
    duration = end - start

    acc = evaluate(global_model)
    if total_params > 0:
        comm_ratio = total_sent / (total_params * num_clients * ROUNDS)
    else:
        comm_ratio = 0

    times.append(duration)
    accuracies.append(acc)
    communications.append(comm_ratio)

    print("Time:", duration)
    print("Accuracy:", acc)
    print("Communication ratio:", comm_ratio)


# -------- Graphs --------

plt.figure()
plt.plot(CLIENT_SETTINGS, times, marker='o')
plt.xlabel("Number of Clients")
plt.ylabel("Training Time (sec)")
plt.title("Time vs Clients")
plt.savefig("./results/time_vs_clients.png")

plt.figure()
plt.plot(CLIENT_SETTINGS, accuracies, marker='o')
plt.xlabel("Number of Clients")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Clients")
plt.savefig("./results/accuracy_vs_clients.png")

plt.figure()
plt.plot(CLIENT_SETTINGS, communications, marker='o')
plt.xlabel("Number of Clients")
plt.ylabel("Communication Ratio")
plt.title("Communication vs Clients")
plt.savefig("./results/communication_vs_clients.png")

print("\nGraphs saved in results/")
