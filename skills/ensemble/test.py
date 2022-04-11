import shutil
import argparse
from pathlib import Path

import pfrl
import numpy as np
from PIL import Image, ImageFont, ImageDraw 

from skills import utils
from skills.option_utils import BaseTrial
from skills.agents.ensemble import EnsembleAgent
from skills.agents.abstract_agent import evaluating


class TestTrial(BaseTrial):
    """
    load the trained agent the step through the envs to see if the Q values and the 
    action taken make sense
    """
    def __init__(self):
        super().__init__()
        args = self.parse_args()
        self.params = self.load_hyperparams(args)
        self.setup()

    def parse_args(self):
        """
        parse the inputted argument
        """
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            parents=[self.get_common_arg_parser()]
        )
        parser.add_argument("--tag", type=str, required=True,
                            help="the experiment_name of the trained agent so we know where to look for loading it")
        args = self.parse_common_args(parser)
        return args

    def check_params_validity(self):
        pass

    def setup(self):
        self.check_params_validity()

        # setting random seeds
        pfrl.utils.set_random_seed(self.params['seed'])

        # get the hyperparams
        hyperparams_file = Path(self.params['results_dir']) / self.params['tag'] / 'hyperparams.csv'
        saved_params = utils.load_hyperparams(hyperparams_file)

        # create the saving directories
        self.saving_dir = Path(self.params['results_dir']).joinpath(self.params['experiment_name'])
        if self.saving_dir.exists():  # remove all existing contents
            shutil.rmtree(self.saving_dir)
        utils.create_log_dir(self.saving_dir)
        self.params['saving_dir'] = self.saving_dir

        # env
        self.env = self.make_env(saved_params['environment'], saved_params['seed'] + 1000)

        # agent
        def phi(x):  # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255
        agent_file = Path(self.params['results_dir']) / self.params['tag'] 
        self.agent = EnsembleAgent(
            device=saved_params['device'],
            warmup_steps=np.inf,  # never update
            batch_size=saved_params['batch_size'],
            phi=phi,
            num_modules=saved_params['num_policies'],
            num_output_classes=self.env.action_space.n,
            action_selection_strategy=saved_params['action_selection_strat'],
        )
        self.agent.load(agent_file)
    
    def run(self):
        """
        start the environment and just execute the trained agent, and see what 
        actions the agent chooses
        """
        obs = self.env.reset()
        step = 0
        action_meanings = self.env.unwrapped.get_action_meanings()
        print(action_meanings)
        with evaluating(self.agent):
            while step < 200:
                a, ensemble_actions, ensemble_q_vals = self.agent.act(obs, return_ensemble_info=True)
                step += 1
                obs, reward, done, info = self.env.step(a)

                # render the image
                frame = np.array(obs)[-1]
                image = Image.fromarray(frame.astype(np.uint8))
                image = pillow_im_add_margin(image, left=100, bottom=70)
                
                # write the action and q value on it too
                meaningful_actions = [action_meanings[i] for i in ensemble_actions]
                meaningful_q_vals = [str(round(q, 2)) for q in ensemble_q_vals]
                txt = "\n".join([a + "  Qval: " + q for a, q in zip(meaningful_actions, meaningful_q_vals)])
                txt += "\n\n taken: " + str(action_meanings[a])
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
                image_editable = ImageDraw.Draw(image)
                image_editable.text((0, 0), txt, 255, font=font)
                image.save(self.saving_dir / f"{step}.png")

                if done:
                    obs = self.env.reset()


def pillow_im_add_margin(pil_img, top=0, right=0, bottom=0, left=0, color=0):
    """
    add margin to a pillow image
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def main():
    trial = TestTrial()
    trial.run()


if __name__ == '__main__':
    main()
