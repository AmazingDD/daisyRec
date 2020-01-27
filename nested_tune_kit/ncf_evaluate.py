import numpy as np
import torch
from tqdm import tqdm


# def ndcg(gt_items, pred_items):
# 	res = 0.
# 	for gt_item in gt_items:
# 		if gt_item in pred_items:
# 			index = pred_items.index(gt_item)
# 			res += np.reciprocal(np.log2(index+2))

# 	return res

def dcg_at_k(r, k):
    assert k >= 1
    r = np.asfarray(r)[:k] != 0
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    assert k >= 1
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def point_metrics(model, test_loader, top_k, test_ur):
	NDCG = []

	for user, item, _ in tqdm(test_loader):
		if torch.cuda.is_available():
			user = user.cuda()
			item = item.cuda()
		else:
			user = user.cpu()
			item = item.cpu()

		predictions = model(user, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		u = set(user.cpu().numpy())
		assert 1<= len(u) < 2, 'candidates number error'
		u = list(u)[0]

		gt_items = list(test_ur[u])

		r = [1 if i in gt_items else 0 for i in recommends]

		# NDCG.append(ndcg(gt_items, recommends))
		NDCG.append(ndcg_at_k(r, top_k))

	# return np.mean(NDCG)
	return NDCG

def pair_metrics(model, test_loader, top_k, test_ur):
	NDCG = []

	for items in tqdm(test_loader):
		user, item = items[0], items[1]
		if torch.cuda.is_available():
			user = user.cuda()
			item = item.cuda()
		else:
			user = user.cpu()
			item = item.cpu()

		predictions, _ = model(user, item, item)
		_, indices = torch.topk(predictions, top_k)
		recommends = torch.take(
				item, indices).cpu().numpy().tolist()

		u = set(user.cpu().numpy())
		assert 1<= len(u) < 2, 'candidates number error'
		u = list(u)[0]

		gt_items = list(test_ur[u])

		r = [1 if i in gt_items else 0 for i in recommends]

		# NDCG.append(ndcg(gt_items, recommends))
		NDCG.append(ndcg_at_k(r, top_k))

	# return np.mean(NDCG)
	return NDCG