import json
import os
from typing import List
import numpy as np
import pandas as pd

# widgets
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

# plots
import matplotlib.pyplot as plt
from plotnine import *
import requests

# stats
import scipy as sp
import statsmodels as sm

import warnings

warnings.filterwarnings("ignore")


class Arm:
    def __init__(self, true_p):
        self.true_p = true_p
        self.reset()

    def reset(self):
        self.impressions = 0
        self.actions = 0

    def get_state(self):
        return self.impressions, self.actions

    def get_rate(self):
        return self.actions / self.impressions if self.impressions > 0 else 0.0

    def pull(self):
        self.impressions += 1
        res = 1 if np.random.random() < self.true_p else 0
        self.actions += res
        return res


class MusketeerEnv:
    def __init__(self, true_ps, avg_impressions):
        self.true_ps = true_ps
        self.avg_impressions = avg_impressions
        self.nb_arms = len(true_ps)
        self.reset()

    def reset(self):
        self.t = -1
        self.ds = []
        self.arms = [Arm(p) for p in self.true_ps]
        return self.get_state()

    def get_state(self):
        return [self.arms[i].get_state() for i in range(self.nb_arms)]

    def get_rates(self):
        return [self.arms[i].get_rate() for i in range(self.nb_arms)]

    # sample the actual number of impressions from a triangular function
    def get_impressions(self):
        return int(
            np.random.triangular(
                self.avg_impressions / 2,
                self.avg_impressions,
                self.avg_impressions * 1.5,
            )
        )

    # ramdomly choose arm based on a given probabiliy `ps`
    def step(self, ps):
        self.t += 1
        impressions = self.get_impressions()
        for i in np.random.choice(a=self.nb_arms, size=impressions, p=ps):
            self.arms[i].pull()
        self.record()
        return self.get_state()

    # for logging
    def record(self):
        d = {"t": self.t, "max_rate": 0, "opt_impressions": 0}

        for i in range(self.nb_arms):
            d[f"impressions_{i}"], d[f"actions_{i}"] = self.arms[i].get_state()
            d[f"rate_{i}"] = self.arms[i].get_rate()

            if d[f"rate_{i}"] > d["max_rate"]:
                d["max_rate"] = d[f"rate_{i}"]
                d["opt_impressions"] = d[f"impressions_{i}"]

        d["total_impressions"] = sum(
            [self.arms[i].impressions for i in range(self.nb_arms)]
        )
        d["opt_impressions_rate"] = d["opt_impressions"] / d["total_impressions"]
        d["total_actions"] = sum([self.arms[i].actions for i in range(self.nb_arms)])
        d["total_rate"] = d["total_actions"] / d["total_impressions"]
        d["regret_rate"] = d["max_rate"] - d["total_rate"]
        d["regret"] = d["regret_rate"] * d["total_impressions"]
        self.ds.append(d)

    # for printting
    def show_df(self):
        df = pd.DataFrame(self.ds)
        cols = (
            ["t"]
            + [f"rate_{i}" for i in range(self.nb_arms)]
            + [f"impressions_{i}" for i in range(self.nb_arms)]
            + [f"actions_{i}" for i in range(self.nb_arms)]
            + ["total_impressions", "total_actions", "total_rate"]
            + ["opt_impressions", "opt_impressions_rate"]
            + ["regret_rate", "regret"]
        )
        df = df[cols]
        return df


# random me a number
import random
import numpy as np


class BanditAgent:
    def __init__(self):
        pass

    @staticmethod
    def get_rate(impressions, actions):
        return actions / impressions if impressions > 0 else 0

    @staticmethod
    def get_rates(state):
        return np.array(
            [__class__.get_rate(impressions, actions) for impressions, actions in state]
        )

    # baselines
    def equal_weights(self, state):
        p_actions = np.array([1 / len(state) for i in range(len(state))])
        return p_actions

    # TODO:1 : write a function that give the probability of choosing arm randomly
    def randomize(self, state):
        random_probs = np.array([random.randint(0, 100) for i in range(len(state))])
        probs_sum = sum(random_probs)
        return random_probs / probs_sum

    # TODO:2 : write a function that give the probability of choosing arm based on epsilon greedy policy
    def eps_greedy(self, state, t, start_eps=0.3, end_eps=0.01, gamma=0.99):
        def get_current_eps(t, start_eps, end_eps, gamma):
            eps_between_start_and_end = start_eps * gamma ** t
            return (
                eps_between_start_and_end
                if eps_between_start_and_end > end_eps
                else end_eps
            )

        rates = __class__.get_rates(state)
        best_arm_index = np.argmax(rates)
        p = random.uniform(0, 1)
        current_eps = get_current_eps(t, start_eps, end_eps, gamma)
        if p > current_eps:
            p_actions = np.zeros(len(state))
            p_actions[best_arm_index] = 1
            return p_actions
        return self.equal_weights(state)

    # TODO:3 : write a function that give the probability of choosing arm based on softmax greedy policy
    def softmax(self, state, t, start_tau=1e-1, end_tau=1e-4, gamma=0.9):
        def get_current_tau(t, start_tau, end_tau, gamma):
            tau_between_start_and_end = start_tau * gamma ** t
            return (
                tau_between_start_and_end
                if tau_between_start_and_end > end_tau
                else end_tau
            )

        rates = __class__.get_rates(state)
        current_tau = get_current_tau(t, start_tau, end_tau, gamma)
        # https://stackoverflow.com/a/42606665
        # prevent overflow of denominator
        max_expo_power = np.max(rates / current_tau)
        normalized_expo_powers = (rates / current_tau) - max_expo_power
        denominator = np.sum(np.exp(normalized_expo_powers))
        return np.exp(normalized_expo_powers) / denominator

    # TODO:4 : write a function that give the probability of choosing arm based on UCB policy
    def ucb(self, state, t):
        selected_arm_index = np.argmax(
            [
                __class__.get_rate(impressions, actions)
                + np.sqrt(2 * np.log(t) / impressions)
                for impressions, actions in state
            ]
        )
        p_actions = np.zeros(len(state))
        p_actions[selected_arm_index] = 1
        return p_actions


# env = MusketeerEnv(true_ps=[0.12, 0.13, 0.15, 0.16], avg_impressions=400)
# a = BanditAgent()
# for i in range(20):
#     p = a.equal_weights(env.get_state())
#     env.step(p)
#     t = i
# print(
#     a.equal_weights(env.get_state()),
#     a.randomize(env.get_state()),
#     a.eps_greedy(env.get_state(), t),
#     a.softmax(env.get_state(), t),
#     a.ucb(env.get_state(), t),
# )

# N_policy = 5
# envs = [
#     MusketeerEnv(true_ps=[0.12, 0.13, 0.15, 0.16], avg_impressions=400)
#     for i in range(N_policy)
# ]
# a = BanditAgent()
# for t in range(250):
#     states = [env.get_state() for env in envs]
#     actions = [
#         a.equal_weights(states[0]),
#         a.randomize(states[1]),
#         a.eps_greedy(states[2], t),
#         a.softmax(states[3], t),
#         a.ucb(states[4], t),
#     ]

#     for i in range(N_policy):
#         # print(i, actions[i])
#         envs[i].step(actions[i])

# dfs = [env.show_df() for env in envs]
# policies = ["equal_weights", "randomize", "eps_greedy", "softmax", "ucb"]
# for i in range(N_policy):
#     dfs[i]["policy"] = policies[i]
# df = pd.concat(dfs)[
#     ["policy", "t", "opt_impressions_rate", "regret_rate", "regret", "total_rate"]
# ]

states_file = "./states.json"


class RealArm:
    token = "9860875610"

    @staticmethod
    def get_id(tweaks: List[int]):
        return "".join(str(t) for t in tweaks)

    def __init__(self, tweaks: List[int], impressions, actions):
        self.tweaks = tweaks
        self.id = __class__.get_id(tweaks)
        self.impressions = impressions
        self.actions = actions

    def get_state(self):
        return self.impressions, self.actions

    def get_rate(self):
        return self.actions / self.impressions if self.impressions > 0 else 0.0

    def save_to_file(self):
        states = None
        with open(states_file, "r", encoding="utf-8") as f:
            states = json.load(f)
        with open(states_file, "w", encoding="utf-8") as f:
            states["arms"][self.id] = {
                "impressions": self.impressions,
                "actions": self.actions,
            }
            states["t"] += 1
            json.dump(states, f, indent=4)

    def pull(self):
        self.impressions += 1
        r = requests.post(
            "https://comengmath.herokuapp.com/update_state",
            json={"times": 1, "arms": self.tweaks},
            headers={"Authorization": __class__.token},
        )
        result = json.loads(r.content)
        self.actions += sum(result["request_reward"])
        self.save_to_file()
        return result

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, RealArm):
            return False
        return self.id == __o.id

    def __repr__(self):
        return f"[{','.join(self.id)}]: {self.impressions=} {self.actions=}"


def get_all_possible_tweaks() -> List[RealArm]:
    arms = []
    combination = [("000000" + ((str(bin(i)))[2:]))[-6:] for i in range(64)]
    tweaks = [1, 2, 3, 4, 5, 6]
    for each in combination:
        selected_tweaks = []
        for tweak, select in zip(tweaks, each):
            if select == "1":
                selected_tweaks.append(tweak)
        arms.append(selected_tweaks)
    return arms


def create_default_states():
    default_states = {}
    default_states["arms"] = {}
    for tweak in get_all_possible_tweaks():
        id = RealArm.get_id(tweak)
        default_states["arms"][id] = {"impressions": 0, "actions": 0}
    default_states["t"] = 0
    return default_states


def get_all_arms(states: dict) -> List[RealArm]:
    arms = []
    arm_states = states["arms"]
    for id, state in arm_states.items():
        arms.append(RealArm(id, state["impressions"], state["actions"]))
    return arms

# def 


def main():
    # load states from file
    # if states file doesn't exist, create it
    if not os.path.exists(states_file):
        with open(states_file, "w", encoding="utf-8") as f:
            json.dump(create_default_states(), f, indent=4)

    states = None
    with open(states_file, encoding="utf-8") as f:
        states = json.load(f)

    arms = get_all_arms(states)
    agent = BanditAgent()

    old_t = states["t"]
    pull_per_arm = 100
    for t in range(old_t, old_t + pull_per_arm):
        probs_select_each_arm = agent.softmax([arm.get_state() for arm in arms], t, 3)
        print(probs_select_each_arm)
        selected_arm = np.random.choice(arms, p=probs_select_each_arm)
        result = selected_arm.pull()
        print(f"{t=} {selected_arm=} {result=}")


if __name__ == "__main__":
    main()
