from gym import Wrapper

from skills.option_utils import get_player_room_number


class MonteObjectGoalWrapper(Wrapper):
    """
    This is the base class for all wrappers that deal with monte moving around some object as a goal
    
    This class is not intended to be used directly, but instead contain common code for specific wrappers
    Currently used by the following wrappers:
        - MonteSkullGoalWrapper
        - MonteSpiderGoalWrapper
        - MonteSnakeGoalWrapper
    """
    def __init__(self, env, epsilon_tol=8):
        """
        Args:
            epsilon_tol: how close to the object the agent must be to finish the skill
        """
        super().__init__(env)
        self.env = env
        self.epsilon_tol = epsilon_tol
        self.room_number = get_player_room_number(self.env.unwrapped.ale.getRAM())
    
    def finished_skill(self, player_x, player_y, object_x, room_number):
        """
        determine if the monte agent has finished the skill
        The agent finishes the skill if the player is:
            - to the left of the object
            - not too far away from the object
            - on the ground
            - in the same room
        """
        on_ground = player_y == self.y
        to_the_left = player_x < object_x and abs(player_x - object_x) < self.epsilon_tol
        in_same_room = room_number == self.room_number
        return on_ground and to_the_left and in_same_room
    