import os
import pickle
import argparse

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--experiment_name', type=str, default='test',
						help='a subdirectory name for the saved results')
	parser.add_argument("--results_dir", type=str, default='results',
                        help='the name of the directory used to store results')
	args = parser.parse_args()
	return args


def plot_experiment_data(experiment_name=None, log_file_name='log_file_0.pkl', results_dir='results'):
	"""
	the single argument is designed solely for the purpose of calling this function
	is __main__.py
	"""
	if experiment_name is None:
		# this is used when main() in ran directly from command line
		args = parse_args()
		experiment_name = args.experiment_name
		results_dir = args.results_dir

	experiment_dir = os.path.join(results_dir, experiment_name, log_file_name)
	plot_learning_curve(experiment_dir)

	img_save_path = os.path.join(results_dir, experiment_name, "learning_curve.png")
	plt.savefig(img_save_path)


def plot_learning_curve(file_path):
	# open logging file
	with open(file_path, 'rb') as f:
		logged_data = pickle.load(f)
	# load data
	time_steps = []
	success_rates = []
	for step in logged_data:
		step_data = logged_data[step]
		if 'success' in step_data:
			time_steps.append(step)
			success_rates.append(step_data['success'])
	# plot
	plt.figure()
	plt.plot(time_steps, success_rates, 'o-')
	plt.title('learning curve')
	plt.xlabel('time step')
	plt.ylabel('success')
	plt.show()


def count(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


def plot_attention_mask_with_original_obs(obs_path, attention_mask_path, save_dir='results'):
	"""
	plot the 3 attentions of ensemble-3, along with a original observation
	from the game screen
	"""
	attention_mask = np.load(attention_mask_path)
	assert len(attention_mask) == 3
	fig, axes = plt.subplots(nrows=2, ncols=2)

	for i, ax in enumerate(axes.flat):
		if i == 0:
			# plot original obs
			original_obs = np.load(obs_path)
			ax.imshow(original_obs)
			ax.set_title('Game screen')
		else:
			# plot the attention masks
			im = ax.imshow(attention_mask[i-1])
			ax.set_title(f"Attention mask {i}")

	fig.subplots_adjust(right=0.8, hspace=0.3)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
	fig.colorbar(im, cax=cbar_ax)

	# save fig
	path = os.path.join(save_dir, 'attention_mask.png')
	plt.savefig(path)


@count
def plot_attention_diversity(
	embedding, 
	num_attentions=8, 
	save_dir=None, 
	plot_freq=1,
	value=None,
	count=None,
	step=None,
	weight=None,
):
	"""
	visualize whether embedding of each attention is getting more and more diverse
	"""
	assert len(embedding) == num_attentions
	# assert embedding[0][0, :, :, :].shape == (64, 10, 10), embedding[0].shape
	embedding_mean = [emb.cpu().detach().numpy().mean(axis=(0, 1)) for emb in embedding]
	for i in range(num_attentions):
		plt.subplot(2, 4, i+1)
		plt.imshow(embedding_mean[i])
		plt.title("attention {}".format(i))
	# show/save fig
	if save_dir is not None:
		if plot_attention_diversity.calls % plot_freq == 0:
			path = os.path.join(save_dir, f"attention_diversity_{plot_attention_diversity.calls}.png")
			plt.savefig(path)
			data_path = os.path.join(save_dir, f"attention_diversity_{plot_attention_diversity.calls}_data.npy")
			np.save(data_path, np.array(embedding_mean))
			value_path = os.path.join(save_dir, f"attention_diversity_{plot_attention_diversity.calls}_reward_values.txt")
			np.savetxt(value_path, value)
			count_path = os.path.join(save_dir, f"attention_diversity_{plot_attention_diversity.calls}_count.txt")
			np.savetxt(count_path, count)
			bandit_values = value + weight * np.sqrt(2 * np.log(step) / count)
			bandit_path = os.path.join(save_dir, f"attention_diversity_{plot_attention_diversity.calls}_bandit_values.txt")
			np.savetxt(bandit_path, bandit_values)
	else:
		plt.show()
	plt.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--attention', '-a', action='store_true', default=True,
						help='plot attention mask')
	parser.add_argument('--load', '-l', type=str,
						help="path to the saved data")
	args = parser.parse_args()

	if args.attention:
		plot_attention_mask_with_original_obs(
			obs_path='results/obs.npy',
			attention_mask_path='results/coinrun/ensemble-3/2/plots/attention_diversity_393_data.npy',
		)
