import json
import os
from typing import Dict, List, Literal, TypedDict, Union
import numpy as np
import requests

# define your BanditAgent class here or import from another file
from BanditAgent import BanditAgent

RewardList = List[Literal[1, 0]]


class MABState(TypedDict):
    count: int
    cumulative_reward: int
    reward_list: RewardList


class GetStatJson(TypedDict):
    state: MABState
    status: int


class ArmState(TypedDict):
    impressions: int
    actions: int


class States(TypedDict):
    arms: Dict[str, ArmState]
    t: int


class PullResponseJson(TypedDict):
    limit_reach: bool
    request_reward: RewardList
    status: int


states_file = "./states.json"


class RealArm:
    token = "9860875610"

    @staticmethod
    def get_stats() -> GetStatJson:
        r = requests.get(
            "https://comengmath.herokuapp.com/get_state",
            headers={"Authorization": __class__.token},
        )
        return r.json()

    @staticmethod
    def get_id(tweaks: List[int]):
        """Get the id from a list of tweaks. e.g. [1,2,3] -> "123" """
        return "".join(str(t) for t in tweaks)

    @staticmethod
    def get_tweaks_from_id(id: str):
        """Get a list of tweaks from an id. e.g. "123" -> [1,2,3]"""
        return [int(e) for e in id]

    def __init__(self, id: str, impressions: int, actions: int):
        """Initialize a new arm with a given id that represents a set of tweaks, impressions and actions.
        Args:
            id: String concatenation of tweaks.
            impressions: Total number of pulls on this arm.
            actions: Total number of rewards on this arm.
        """
        self.id = id
        self.tweaks = __class__.get_tweaks_from_id(id)
        self.impressions = impressions
        self.actions = actions

    def get_state(self):
        return self.impressions, self.actions

    def get_rate(self):
        return self.actions / self.impressions if self.impressions > 0 else 0.0

    def save_to_file(self):
        """Save the current state of the arm to the file."""
        states: States = None
        with open(states_file, "r", encoding="utf-8") as f:
            states = json.load(f)
        with open(states_file, "w", encoding="utf-8") as f:
            states["arms"][self.id] = {
                "impressions": self.impressions,
                "actions": self.actions,
            }
            states["t"] += 1
            json.dump(states, f, indent=4)

    def pull(self, times: int = 1) -> PullResponseJson:
        """Pull the arm for a given number of times defaulted to 1."""
        self.impressions += times
        r = requests.post(
            "https://comengmath.herokuapp.com/update_state",
            json={"times": times, "arms": self.tweaks},
            headers={"Authorization": __class__.token},
        )
        result: PullResponseJson = json.loads(r.content)
        self.actions += sum(result["request_reward"])
        self.save_to_file()
        return result

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, RealArm):
            return False
        return self.id == __o.id

    def __repr__(self):
        return f"{self.tweaks}: rate={self.get_rate():.3f} impressions={self.impressions} actions={self.actions}"


def get_all_possible_tweaks():
    number_of_arms = 64
    tweak_combinations: List[List[int]] = []
    tweak_enable_combinations = [
        ("000000" + ((str(bin(i)))[2:]))[-6:] for i in range(number_of_arms)
    ]
    tweak_options = [1, 2, 3, 4, 5, 6]
    for enable in tweak_enable_combinations:
        enabled_tweaks = [
            tweak for tweak, select in zip(tweak_options, enable) if select == "1"
        ]
        tweak_combinations.append(enabled_tweaks)
    return tweak_combinations


def create_default_states():
    default_states: States = {"arms": {}, "t": 0}
    for tweak in get_all_possible_tweaks():
        id = RealArm.get_id(tweak)
        default_states["arms"][id] = {"impressions": 0, "actions": 0}
    return default_states


def get_all_arms(states: States) -> List[RealArm]:
    arm_states = states["arms"]
    return [
        RealArm(id, state["impressions"], state["actions"])
        for id, state in arm_states.items()
    ]


def print_probs(
    arms: List[RealArm], probs_select_each_arm: List[float], limit: int = None
):
    """Print the probabilities of being selected in the next pull from high to low of each arm with. If limit is specified, only print the first limit arms."""
    print_count = 0
    print("Probabilities of being selected of each arm in the next pull")
    for arm, prob in sorted(
        zip(arms, probs_select_each_arm), key=lambda e: e[0].get_rate(), reverse=True
    ):
        print(f"{prob=:.5f} {arm=}")
        print_count += 1
        if limit and print_count == limit:
            break


def select_specific_arm(tweaks: List[int], arms: List[RealArm]) -> Union[RealArm, None]:
    """Find and return the arm with the given tweaks. If no arm is found, return None."""
    return next(filter(lambda arm: arm.tweaks == tweaks, arms), None)


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
    start_t = old_t + 1
    pulls_per_arm = 1
    for t in range(start_t, start_t + pulls_per_arm):
        states = [arm.get_state() for arm in arms]

        ## select an arm based on MAB policy
        # probs_select_each_arm = agent.softmax(states, t, 2, 0.01)
        # probs_select_each_arm = agent.equal_weights(states)
        # probs_select_each_arm = agent.eps_greedy(states, t, 0.5, 0.4)
        probs_select_each_arm = agent.ucb(states, t)
        # selected_arm = np.random.choice(arms, p=probs_select_each_arm)

        ## select a specific arm
        # selected_arm = select_specific_arm([2, 6], arms)

        ## show the probabilities of each arm being selected in the next pull
        print_probs(arms, probs_select_each_arm, limit=5)

        ## pull the selected arm
        # print(f"{t=} {selected_arm=} ", end="")
        # result = selected_arm.pull()
        # print(f"{result=}")

    ## print the overall statistics of MAB
    # print(RealArm.get_stats())

    # for arm in sorted(arms, key=lambda a: a.get_rate()):
    #     print(arm)


if __name__ == "__main__":
    main()
