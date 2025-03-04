"""Test"""

from json import dumps
import logging

from src.position_counter_automaton import PositionCounterAutomaton
from src.position_counter_automaton import super_config_to_json


def main() -> None:
    pattern = input()
    automaton = PositionCounterAutomaton.create(pattern)
    print(automaton)
    while True:
        text = input()
        if text == "END":
            break

        # super_congis = automaton.iterate_super_configs(text)
        # counter_cartesian_super_configs = (
        #     automaton.iterate_counter_cartesian_super_configs(text)
        # )
        # for super_config, counter_cartesian_super_config in zip(
        #     super_congis, counter_cartesian_super_configs
        # ):
        #     print(dumps(super_config_to_json(super_config)))
        #     print(dumps(counter_cartesian_super_config.unfold()))

        #     print(dumps(counter_cartesian_super_config.to_json()))
        #     print()
        super_congis = automaton.iterate_super_configs(text)
        for super_config in super_congis:
            print(dumps(super_config_to_json(super_config)))
        #     print(dumps(counter_cartesian_super_config.to_json()))
        #     print()
        print(automaton(text))

if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    main()
