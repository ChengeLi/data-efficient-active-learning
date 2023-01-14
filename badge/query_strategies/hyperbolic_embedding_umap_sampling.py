import numpy as np
import torch
from .strategy import Strategy

class UmapHBSampling(Strategy):
	def __init__(self, X, Y, idxs_lb, net, handler, args):
		super(UmapHBSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

	def query(self, n):
		# TODO: Implement UMAP and Hyperbolic projection
		pass
