import numpy as np
import random


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
