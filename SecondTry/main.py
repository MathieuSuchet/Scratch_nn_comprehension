import warnings
from abc import ABC
from typing import Optional, Iterable, Type, Tuple

import gymnasium as gym
import pygame.event
from gymnasium import Env

if __name__ == "__main__":

    # Level 1 : Basic env rendering
    # - Environment rendering
    # - Verbose
    # - Environment live-info
    # - Agent live-info

    def render_env(env_name: str,
                   verbose: int = 1,
                   display_env_returns: bool = False,
                   display_agent_returns: bool = False):
        """
        Render an environment with a random agent playing

        :param env_name: Name of the environment to render
        :param verbose: 0 for no info at runtime, 1 for basic info, 2 for interruptions
        :param display_env_returns: True to display what the environment returns (might affect performance)
        :param display_agent_returns: True to display what the agent returns (might affect performance)
        """

        env = gym.make(env_name, render_mode="human")

        old_verbose = verbose
        verbose = min(2, verbose)
        verbose = max(0, verbose)

        if old_verbose != verbose:
            warnings.warn(f"Verbose has to be between 0 and 2 (both included), given verbose was {old_verbose}"
                          f", verbose considered as {verbose}", RuntimeWarning)

        if verbose >= 1:
            print(f"{env_name} environment created")

        action_shape = env.action_space.n
        observation_shape = env.observation_space.shape[0]

        if verbose >= 1:
            print(f"Action size : {action_shape}")
            print(f"Observation size : {observation_shape}")

        obs = env.reset()

        while True:
            try:
                env.render()

                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                if display_env_returns:
                    print(f"\n=====New step=====")
                    print(f"Agent action : {action}") if display_agent_returns else None
                    print(f"New state    : {obs}")
                    print(f"Reward       : {reward}")
                    print(f"Terminated   : {terminated}")
                    print(f"Truncated    : {truncated}")
                    print(f"Info         : {info}")
                else:
                    if display_agent_returns:
                        print(f"\n=====New step=====")
                        print(f"Agent action : {action}")

                if terminated or truncated:
                    obs = env.reset()

                    if terminated and verbose >= 1:
                        print("Goal reached, resetting...")

                    if truncated and verbose >= 1:
                        print("Episode reached 200 steps, resetting...")

                quitEvents = pygame.event.get(pygame.QUIT)
                if len(quitEvents) > 0:
                    if verbose >= 2:
                        print("Window quit called, exiting...")
                    break

            except KeyboardInterrupt:
                if verbose >= 2:
                    print("User interruption detected, exiting...")
                break


    # Level 2 : Dividing in modules
    # Environment rendering made in a different class
    # Agent actions made in a different class
    # Termination/Truncation check done in a different class

    class MissingEnvError(Exception):
        """
        Error thrown when an environment is not specified
        """
        pass


    class Renderer(ABC):
        def __init__(self, env: Env):
            self.env = env

        def render(self):
            """
            Method used to render the environment
            :raise MissingEnvError: Environment is not specified
            """
            if self.env is None:
                raise MissingEnvError("No environment given to the renderer")

            self.env.render()


    class Agent(ABC):
        def __init__(self, env: Env):
            self.env = env
            self.action_size = self.env.action_space.n
            self.observation_size = self.env.observation_space.shape[0]

        def act(self, obs) -> int:
            """
            Return the action made by the agent based on the given observation

            :param obs: Observation returned by the env
            :raise MissingEnvError: Environment is not specified
            :return: The action made by the agent
            """
            if self.env is None:
                raise MissingEnvError("No environment given to the agent")

            return self.env.action_space.sample()


    class EndCheck(ABC):

        def __init__(self, env: Env):
            self.env = env

        def check_end(self, terminated: bool, truncated: bool, verbose: int = 1) -> Tuple[bool, Iterable]:
            """
            Check whether the observation is terminated or truncated

            :param terminated: State termination
            :param truncated: State truncation
            :param verbose: 0 for no info, 1 for end info
            :return: Boolean for environment reset (True if reset, False otherwise) and the obs if reset
            """

            if terminated or truncated:

                if terminated and verbose >= 1:
                    print("Goal reached, resetting...")

                if truncated and verbose >= 1:
                    print("Episode reached 200 steps, resetting...")

                return True, self.env.reset()
            return False, []

    class StupidEndCheck(EndCheck):
        def check_end(self, terminated: bool, truncated: bool, verbose: int = 1) -> Tuple[bool, Iterable]:
            print("Nah bru, no prahblem")
            return False, []


    def render_env_divided(env_name: str,
                           verbose: int = 1,
                           display_env_returns: bool = False,
                           display_agent_returns: bool = False,
                           renderer: Optional[Type[Renderer]] = None,
                           agent: Optional[Type[Agent]] = None,
                           end_check: Optional[Type[EndCheck]] = None,
                           ):
        """
        Render an environment with a random agent playing (divided)

        :param env_name: Name of the environment to render
        :param verbose: 0 for no info at runtime, 1 for basic info, 2 for interruptions
        :param display_env_returns: True to display what the environment returns (might affect performance)
        :param display_agent_returns: True to display what the agent returns (might affect performance)
        :param renderer: Type that will run the environment rendering
        :param agent: Type that will run the agent acting
        :param end_check: Type that will run the termination/truncation testing

        """

        env = gym.make(env_name, render_mode="human")

        old_verbose = verbose
        verbose = min(2, verbose)
        verbose = max(0, verbose)

        if old_verbose != verbose:
            warnings.warn(f"Verbose has to be between 0 and 2 (both included), given verbose was {old_verbose}"
                          f", verbose considered as {verbose}", RuntimeWarning)

        if verbose >= 1:
            print(f"{env_name} environment created")

        action_shape = env.action_space.n
        observation_shape = env.observation_space.shape[0]

        if verbose >= 1:
            print(f"Action size : {action_shape}")
            print(f"Observation size : {observation_shape}")

        obs = env.reset()

        renderer = Renderer(env) if not renderer else renderer(env)
        agent = Agent(env) if not agent else agent
        end_check = EndCheck(env) if not end_check else end_check

        while True:
            try:
                renderer.render()

                action = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)

                if display_env_returns:
                    print(f"\n=====New step=====")
                    print(f"Agent action : {action}") if display_agent_returns else None
                    print(f"New state    : {obs}")
                    print(f"Reward       : {reward}")
                    print(f"Terminated   : {terminated}")
                    print(f"Truncated    : {truncated}")
                    print(f"Info         : {info}")
                else:
                    if display_agent_returns:
                        print(f"\n=====New step=====")
                        print(f"Agent action : {action}")

                ended, new_obs = end_check.check_end(terminated, truncated, verbose)

                if ended:
                    obs = new_obs

                quitEvents = pygame.event.get(pygame.QUIT)
                if len(quitEvents) > 0:
                    if verbose >= 2:
                        print("Window quit called, exiting...")
                    break

            except KeyboardInterrupt:
                if verbose >= 2:
                    print("User interruption detected, exiting...")
                break


    render_env_divided(env_name="CartPole-v1",
                       verbose=3,
                       display_env_returns=False,
                       display_agent_returns=False)
