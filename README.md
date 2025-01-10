# Prize-Driven Multi-Agent Reinforcement Learning (P-MARL) for Budget-Constrained Traveling Salesman Problem (BC-TSP) 

This repository hosts the source code for the **P-MARL algorithm** featured in the paper: **"Budget-Constrained Traveling Salesman Problem: a Cooperative Multi-Agent Reinforcement Learning Approach,"** published in the proceedings of **IEEE SECON, December 2024**.

## Overview:
  * This research addresses the **BC-TSP**, a variation of the classic Traveling Salesman Problem where the goal is to **maximize prize collection** within a given budget.
  * **P-MARL** is a novel **multi-agent reinforcement learning framework** designed to efficiently solve the BC-TSP.
  * P-MARL leverages the **synergy between prize maximization** in BC-TSP and **cumulative reward maximization in RL**.
  * The algorithm utilizes a **hybrid approach** with **independent and cooperative learning phases** for enhanced prize collection.
  * P-MARL integrates **node prizes** into the reward model and action mechanism, guiding agents towards optimal prize-collecting paths.

## Key Features:
  * **Prize-Based Action Mechanism:** Agents select actions based on exploitation, exploration, or termination strategies that consider node prizes, Q-values, and edge weights.
  * **Independent Update of PC-Table:** Agents independently update the Prize-Collecting Table (PC-Table) based on their experiences, contributing to a distributed learning process.
  * **Prize-Based Reward Model:** Agents collaboratively identify the maximum-prize route and update reward values and Q-values for edges on this route, promoting cooperative learning.
  * **Two-Stage Approach:** P-MARL involves a learning stage where agents build the PC-Table and an execution stage where the salesman follows the PC-Table for prize collection.
  * **Efficiency:** P-MARL is significantly faster than deep reinforcement learning approaches, taking two orders of magnitude less training time.

## Performance:
  * Extensive simulations using real-world data (48 US state capital cities) show P-MARL's superior performance.
  * P-MARL **outperforms** existing prize-oblivious MARL (Ant-Q), handcrafted greedy algorithms, and state-of-the-art DRL (RNN).
  * P-MARL achieves near-optimal results compared to the optimal ILP solution.
  * Demonstrates **strong prize collection capabilities** while staying within budget constraints.

## Usage:
  * The code is well-documented and provides instructions for running simulations and experiments.
  * Users can modify parameters and datasets to explore various scenarios and applications.

## Requirements:
  * Programming language: Python
  * Deep Learning libraries: PyTorch 
  * Other dependencies: (Specify any additional libraries or packages required)

## Future Work:
  * Convergence study of the P-MARL algorithm.
  * Explore applications of BC-TSP and P-MARL in diverse domains. 
  * Investigate extensions of P-MARL for handling dynamic environments and uncertainties.

##  Contributions:
This work is a collaborative effort by researchers at California State University Dominguez Hills and California State University, Long Beach. 

This repository provides researchers and practitioners with a valuable tool for understanding and implementing the P-MARL algorithm for solving the BC-TSP.  
