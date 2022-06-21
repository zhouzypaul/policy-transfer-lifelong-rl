from gym import Wrapper

from skills.option_utils import get_player_position, get_object_position, get_player_room_number


# y pos of the player in those rooms on the platform
room_to_y = {
    4: 235,
    13: 235, 
    21: 235,
}


class MonteSpiderGoalWrapper(Wrapper):
    """
    for training a "jump over spider" skill.
    The agent finishes the skill if its y pos aligns with the floor of the spider and 
    its x pos is on the other side of the spider.

    currently, default to the player starts on the right side of the spider, and try to jump to the left of it
    """
    def __init__(self, env, epsilon_tol=6):
        """
        Args:
            epsilon_tol: the agent dies within 6 of the spider
        """
        super().__init__(env)
        self.env = env
        self.epsilon_tol = epsilon_tol
        self.room_number = get_player_room_number(self.env.unwrapped.ale.getRAM())
        self.y = room_to_y[self.room_number]
    
    def step(self, action):
        """
        override done and reward
        """
        next_state, reward, done, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM()
        room = get_player_room_number(ram)
        player_x, player_y = get_player_position(ram)
        spider_x, spider_y = get_object_position(ram)
        if player_y == self.y and player_x < spider_x - self.epsilon_tol and room == self.room_number:
            done = True
            reward = 1
        else:
            reward = 0  # override reward, such as when got key
        # terminate if agent enters another room
        if room != self.room_number:
            done = True
        # override needs_real_reset for EpisodicLifeEnv
        self.env.unwrapped.needs_real_reset = done or info.get("needs_reset", False)
        return next_state, reward, done, info
