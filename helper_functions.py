import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import Tensor


def plot_decision_boundary(model: nn.Module, X: Tensor, y: Tensor, num_points: int = 250):
	"""
	Plots the decision boundary of a pytorch model for 2D data.
	Detects binary or multiclass classification from the labels.

	args:
		model: pytorch (trained) model
		X: input features with shape (n_samples, n_features)
		y: target labels as a vector of size n_samples

	"""

	first_parameter = next(model.parameters(), None)
	if first_parameter is None:
		raise ValueError('The input model has no parameters!')

	current_device = first_parameter.device

	# move the model and the tensors to cpu
	model.to('cpu')
	X, y = X.to('cpu'), y.to('cpu')

	# create a grid of points to cover the features plane
	feature1_min, feature1_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
	feature2_min, feature2_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2

	grid_feature1, grid_feature2 = np.meshgrid(
		np.linspace(feature1_min, feature1_max, num_points),
		np.linspace(feature2_min, feature2_max, num_points),
	)

	grid_points = np.column_stack(
		(grid_feature1.ravel(), grid_feature2.ravel())
	)

	grid_points = torch.from_numpy(grid_points).float()


	# get the number of classes(labels) to detect binary/multiclass classificatoin
	num_labels = len(torch.unique(y))


	# make predictions
	model.eval()
	with torch.inference_mode():
		raw_preds = model(grid_points)

		# get the number of output neurons
		num_output_neurons = raw_preds.shape[1]

		# seperate binary vs. multiclass
		if num_labels == 2:
			# binary
			if num_output_neurons == 1:
				# binary classification with sigmoid
				probabilities = torch.sigmoid(raw_preds).squeeze()
				pred_classes = (probabilities > 0.5).long()
			elif num_output_neurons == 2:
				# binary classification with softmax
				# apply the softmax row-wise -> dim=1
				probabilities = torch.softmax(raw_preds, dim=1)
				pred_classes = probabilities.argmax(dim=1)
			else:
				raise ValueError(
					'Binary classification requires 1 or 2 output neurons! '
					f'But model has {num_output_neurons}.'
				)
		else:
			# multiclass
			if num_output_neurons != num_labels:
				raise ValueError(
					f'Model has {num_output_neurons} output neurons, but data has {num_labels} classes. '
					'For multiclass classification these two should match.'
				)

			# apply the softmax row-wise -> dim=1
			probabilities = torch.softmax(raw_preds, dim=1)
			pred_classes = probabilities.argmax(dim=1)


	boundaries = pred_classes.reshape(grid_feature1.shape)

	plt.figure(figsize=(8, 8))

	plt.contourf(grid_feature1, grid_feature2, boundaries, cmap='RdYlBu', alpha=0.4)

	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, edgecolors='w', cmap='RdYlBu')

	plt.xlabel('Feature 1')
	plt.ylabel('Feature 2')
	plt.title('Decision Boundary')

	plt.show()


	# put the model back on the previous device
	model.to(current_device)


def main():
	torch.manual_seed(23)

	model = nn.Sequential(
		nn.Linear(2, 6),
		nn.ReLU(),
		nn.Linear(6, 1),
	)


	X = torch.row_stack(
		(torch.randint(0, 5, (50, 2)), torch.randint(-4, 1, (50, 2)))
	).float()

	y = torch.row_stack((torch.zeros((50, 1)), torch.ones((50, 1)))).float()


	#loss_fn = nn.BCEWithLogitsLoss()
	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

	for epoch in range(8000):
		model.train()

		y_logits = model(X)
		#y_preds = torch.round(torch.sigmoid(y_logits))
		y_probs = torch.sigmoid(y_logits)

		loss = loss_fn(y_probs, y)
		#loss = loss_fn(y_logits, y)


		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		if epoch % 200 == 0:
			print(f'{loss=:.2f}')

	plot_decision_boundary(model, X, y)


if __name__ == '__main__':
	main()
