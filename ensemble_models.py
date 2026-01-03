import numpy as np
from collections import Counter

# 1. DECISION TREE (Base Learner cho Random Forest & Voting)
def tinh_gini(y):
    m = len(y)
    if m == 0:
        return 0
    dem = np.bincount(y)
    prob = dem / m
    return 1 - np.sum(prob**2)

def tim_cat_tot_nhat(X, y):
    m, n = X.shape
    if m <= 1:
        return None, None
    gini_goc = tinh_gini(y)
    best_gain = -1
    best_split = None
    for cot in range(n):
        nguong_vals = np.unique(X[:, cot])
        for nguong in nguong_vals:
            idx_trai = np.where(X[:, cot] <= nguong)[0]
            idx_phai = np.where(X[:, cot] > nguong)[0]
            if len(idx_trai) == 0 or len(idx_phai) == 0:
                continue
            n_l, n_r = len(idx_trai), len(idx_phai)
            gini_con = (n_l/m)*tinh_gini(y[idx_trai]) + (n_r/m)*tinh_gini(y[idx_phai])
            gain = gini_goc - gini_con
            if gain > best_gain:
                best_gain = gain
                best_split = (cot, nguong)
    return best_gain, best_split

def xay_cay(X, y, do_sau=0, max_depth=10):
    n_labels = len(np.unique(y))
    if do_sau >= max_depth or n_labels == 1:
        return {'loai': 'la', 'kq': Counter(y).most_common(1)[0][0]}
    gain, split = tim_cat_tot_nhat(X, y)
    if gain is None or gain == 0:
        return {'loai': 'la', 'kq': Counter(y).most_common(1)[0][0]}
    cot, nguong = split
    idx_trai = np.where(X[:, cot] <= nguong)[0]
    idx_phai = np.where(X[:, cot] > nguong)[0]
    return {
        'loai': 're_nhanh',
        'cot_idx': cot,
        'nguong': nguong,
        'nhanh_trai': xay_cay(X[idx_trai], y[idx_trai], do_sau+1, max_depth),
        'nhanh_phai': xay_cay(X[idx_phai], y[idx_phai], do_sau+1, max_depth)
    }

def du_doan_cay(node, x):
    if node['loai'] == 'la':
        return node['kq']
    if x[node['cot_idx']] <= node['nguong']:
        return du_doan_cay(node['nhanh_trai'], x)
    return du_doan_cay(node['nhanh_phai'], x)

# 2. K-NEAREST NEIGHBORS (cho Voting Classifier)
def chay_knn(X_train, y_train, X_test, k=5):
    ket_qua = []
    for x in X_test:
        dist = np.sqrt(np.sum((X_train - x)**2, axis=1))
        k_idx = np.argsort(dist)[:k]
        k_nhan = y_train[k_idx]
        ket_qua.append(Counter(k_nhan).most_common(1)[0][0])
    return np.array(ket_qua)

# 3. LOGISTIC REGRESSION (cho Voting Classifier)
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  

def train_logistic(X, y, lr=0.1, n_iters=1000):
    cac_lop = np.unique(y)
    models = []
    for cls in cac_lop:
        y_bin = np.where(y == cls, 1, 0)
        w = np.zeros(X.shape[1])
        b = 0
        for _ in range(n_iters):
            y_pred = sigmoid(np.dot(X, w) + b)
            dw = (1/len(X)) * np.dot(X.T, (y_pred - y_bin))
            db = (1/len(X)) * np.sum(y_pred - y_bin)
            w -= lr * dw
            b -= lr * db
        models.append((cls, w, b))
    return models

def predict_logistic(models, X):
    preds = []
    for x in X:
        scores = [sigmoid(np.dot(x, w) + b) for _, w, b in models]
        preds.append(models[np.argmax(scores)][0])
    return np.array(preds)

# 4. RANDOM FOREST (BAGGING)
class RandomForest:
    def __init__(self, n_trees=20, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.rung_cay = []
    
    def fit(self, X, y):
        self.rung_cay = []
        n_mau = X.shape[0]
        for i in range(self.n_trees):
            idx = np.random.choice(n_mau, n_mau, replace=True)
            cay = xay_cay(X[idx], y[idx], max_depth=self.max_depth)
            self.rung_cay.append(cay)
        return self
    
    def predict(self, X):
        predictions = []
        for x in X:
            votes = [du_doan_cay(cay, x) for cay in self.rung_cay]
            predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(predictions)
    
    def predict_proba(self, X):
        n_classes = 3 
        probas = []
        for x in X:
            votes = [du_doan_cay(cay, x) for cay in self.rung_cay]
            vote_counts = Counter(votes)
            proba = np.zeros(n_classes)
            for cls, count in vote_counts.items():
                proba[cls] = count / self.n_trees
            probas.append(proba)
        return np.array(probas)

# 5. VOTING CLASSIFIER
class VotingClassifier:
    def __init__(self):
        self.tree = None
        self.lr_models = None
        self.X_train_knn = None
        self.y_train_knn = None
    
    def fit(self, X, y):
        print("Đang chạy Decision Tree")
        self.tree = xay_cay(X, y, max_depth=5)  
        print("Đang chạy Logistic Regression")
        self.lr_models = train_logistic(X, y, lr=0.1, n_iters=1000)
        print("Đang chạy KNN")
        self.X_train_knn = X
        self.y_train_knn = y
        return self
    
    def predict(self, X):
        dt_preds = [du_doan_cay(self.tree, x) for x in X]
        lr_preds = predict_logistic(self.lr_models, X)
        knn_preds = chay_knn(self.X_train_knn, self.y_train_knn, X, k=5)
        voting_preds = []
        for i in range(len(X)):
            phieu = [dt_preds[i], lr_preds[i], knn_preds[i]]
            voting_preds.append(Counter(phieu).most_common(1)[0][0])
        return np.array(voting_preds)
    
    def predict_proba(self, X):
        n_classes = 3
        probas = []
        for i in range(len(X)):
            x = X[i:i+1]
            dt_pred = du_doan_cay(self.tree, X[i])
            lr_pred = predict_logistic(self.lr_models, x)[0]
            knn_pred = chay_knn(self.X_train_knn, self.y_train_knn, x, k=5)[0]
            votes = [dt_pred, lr_pred, knn_pred]
            vote_counts = Counter(votes)
            proba = np.zeros(n_classes)
            for cls, count in vote_counts.items():
                proba[cls] = count / 3
            probas.append(proba)
        return np.array(probas)

# 6. XGBOOST CUSTOM (GRADIENT BOOSTING)
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:  
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if self.n_features is None else min(self.n_features, X.shape[1])
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._build_tree(X, y)
        return self
    
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=np.mean(y))
        best_feat, best_thresh, best_gain = self._best_split(X, y)
        if best_feat is None:
            return Node(value=np.mean(y))
        self.feature_importances_[best_feat] += best_gain
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs
        left_child = self._build_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, 
                   left=left_child, right=right_child)
    
    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None
        current_uncertainty = np.var(y) * len(y)
        for feat_idx in range(self.n_features if self.n_features else X.shape[1]):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)    
            for threshold in thresholds:
                left_idxs = X_column <= threshold
                if np.sum(left_idxs) == 0 or np.sum(~left_idxs) == 0:
                    continue
                n_l, n_r = np.sum(left_idxs), np.sum(~left_idxs)
                var_l, var_r = np.var(y[left_idxs]), np.var(y[~left_idxs])
                child_uncertainty = n_l * var_l + n_r * var_r
                gain = current_uncertainty - child_uncertainty
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold       
        return split_idx, split_thresh, best_gain
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class XGBoostCustom:
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3, min_samples_split=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.base_pred = None
        self.feature_importances_ = None
        self.init_pred = None
    
    def fit(self, X, y):
        self.trees = []
        self.init_pred = np.mean(y)
        self.base_pred = self.init_pred
        F = np.full(len(y), self.init_pred, dtype=float)
        for i in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            F += self.learning_rate * predictions
            self.trees.append(tree)
        if self.trees:
            self.feature_importances_ = np.zeros(X.shape[1])
            for tree in self.trees:
                if tree.feature_importances_ is not None:
                    self.feature_importances_ += tree.feature_importances_
            self.feature_importances_ /= len(self.trees)
        return self
    
    def predict(self, X):
        bias = 0.0
        if getattr(self, 'init_pred', None) is not None:
            bias = self.init_pred
        elif self.base_pred is not None:
            bias = self.base_pred
        y_pred = np.full(len(X), bias, dtype=float)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        predictions = np.round(y_pred).astype(int)
        predictions = np.clip(predictions, 0, 2)
        return predictions
    
    def predict_proba(self, X):
        bias = 0.0
        if getattr(self, 'init_pred', None) is not None:
            bias = self.init_pred
        elif self.base_pred is not None:
            bias = self.base_pred   
        F = np.full(len(X), bias, dtype=float)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        n_classes = 3
        probas = []
        for f in F:
            distances = np.abs(f - np.array([0, 1, 2]))
            probs = 1 / (distances + 0.1)
            probs /= probs.sum()
            probas.append(probs)
        return np.array(probas)

# 7. ADABOOST (MULTICLASS)

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None
    
    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = 1
            predictions[X_column >= self.threshold] = -1
        return predictions


class AdaBoostBinary:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.clfs = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))
        self.clfs = []
        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1
                    misclassified = w[y != predictions]
                    error = sum(misclassified)
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_idx = feature_i
                        min_error = error
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y * predictions)
            w /= np.sum(w)  
            self.clfs.append(clf) 
        return self
    
    def decision_function(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.clfs]
        return np.sum(clf_preds, axis=0)
    
    def predict(self, X):
        decision_scores = self.decision_function(X)
        return np.where(decision_scores < 0, 0, 1)


class AdaBoostMulticlass:
    def __init__(self, n_clf=50):
        self.n_clf = n_clf
        self.models = [] 
        self.classes = []  
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = [] 
        print(f"Training {len(self.classes)} binary classifiers")
        for idx, cls in enumerate(self.classes):
            y_binary = np.where(y == cls, 1, -1)
            model = AdaBoostBinary(n_clf=self.n_clf)
            model.fit(X, y_binary)
            self.models.append(model)
            print(f"Classifier {idx+1}/{len(self.classes)} trained")
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        if not self.models:
            return np.zeros(n_samples, dtype=int)
        scores = np.zeros((n_samples, len(self.classes)))
        for idx, model in enumerate(self.models):
            scores[:, idx] = model.decision_function(X)
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        scores = np.zeros((n_samples, n_classes))
        for idx, model in enumerate(self.models):
            scores[:, idx] = model.decision_function(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probas

# 8. UTILITY FUNCTIONS
def tinh_accuracy(y_that, y_du_doan):
    dung = np.sum(y_that == y_du_doan)
    return dung / len(y_that)