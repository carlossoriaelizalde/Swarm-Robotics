# 🤖 Operative Robots in Store

This repository contains a **simulation framework** for autonomous cooperative robots operating inside a **warehouse environment**, where they collaboratively fulfill customer orders and optimize logistics through **swarm intelligence** and **genetic algorithms**.

---

## 📦 Project Overview

The project simulates a logistics center where multiple robots:
- Retrieve products from storage boxes to fulfill client demands.
- Share optimized routes and experience (swarm learning).
- Reduce unnecessary computational effort by discarding non-optimal routes (based on experience)
- Periodically reorganize the warehouse using a **Genetic Algorithm (GA)** to minimize travel distances.

At the beginning of the simulation, the store is divided into two isolated sections (East/West). Later, the barrier is removed, allowing full cooperation between robots and cross-learning.

---

## 🧩 Core Components

- **Store Class**  
  Defines the environment: boxes, stalls, clients, and simulation grid.  
  Handles initialization, positioning, and probabilistic order generation.

- **Robot Class**  
  Represents each autonomous robot.  
  Handles client orders, optimal routing (via Dijkstra), interaction with other robots, and execution of GA-based reorganization.

- **Client Class** 
  Generates customer requests according to configurable probability distributions.

---

## 🧠 Key Algorithms & Concepts

- **Pathfinding:** Dijkstra algorithm with spatial constraints.  
- **Swarm communication:** robots share optimal routes and performance metrics.  
- **Optimization:** Genetic Algorithm (`deap` library) to reorganize box layout and improve future performance.  
- **Simulation Engine:** `simpy` for discrete event simulation and `matplotlib` for visualization.

---

## ⚙️ Requirements

Python 3.11 or higher  
Install dependencies with:

conda env create -f requirements.yml
conda activate swarm-warehouse

## 📊 Outputs

- Step-by-step **matplotlib animations** of warehouse layout and robot movement.
  
- GA **performance metrics** per generation.
  
- Comparative evaluation of configurations (before/after merge, per-section optimization).

## 🎓 Research Context

This project was developed within the **Swarm Robotics Research Grant (2023–2024)** at the **Public University of Navarre (UPNA)**, under the supervision of **Prof. José Enrique Armendáriz-Íñigo**.

The objective of the research was to **design distributed systems combining statistical reasoning, optimization, and machine learning for collective decision-making** in robotic environments.

The warehouse simulation serves as a testbed for evaluating decentralized optimization strategies and **emergent cooperation dynamics among agents**. The framework is currently being extended toward a research paper exploring hybrid approaches that integrate metaheuristics for real-time adaptation in dynamic environments.

## 🧑‍💻 Author
**Carlos Soria Elizalde**

## 📜 License
**MIT**

## 📞 Contact
[carlosoriaeli@gmail.com]

