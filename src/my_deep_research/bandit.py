import numpy as np
import json
import os

BANDIT_PATH = "./bandit_model.json"

class LinUCB:
    def __init__(self,n_actions=3,n_features=3,alpha=1.5):
        self.n_actions = n_actions
        self.n_features = n_features
        self.alpha = alpha
        self.total_updates = 0

        # 3 个动作，每个动作一个 3×3 单位矩阵
        self.A = [np.eye(n_features) for _ in range(n_actions)]

        # 3 个动作，每个动作一个长度为 3 的零向量
        self.b = [np.zeros(n_features) for _ in range(n_actions)]

    def select_action(self, features):
        # 存每个动作的 UCB 分数
        scores = []
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            # 用权重预测这个动作的奖励
            predicted = features @ theta
            # 探索加成。这个动作数据越少，A_inv 越大，bonus 越高，越倾向被选中
            bonus = self.alpha * np.sqrt(features @ A_inv @ features)
            scores.append(predicted + bonus)
        # 选分数最高的动作索引（0=local, 1=web, 2=both）
        return int(np.argmax(scores))
    
    def update(self, action, features, reward):
        # 更新计数
        self.total_updates += 1
        # 用这个动作的特征更新对应矩阵
        self.A[action] += np.outer(features, features)
        self.b[action] += reward * features


_bandit = None

def _get_bandit():
    global _bandit
    # 如果内存里已有实例 → 直接返回
    if _bandit is not None:
        return _bandit
    # 如果文件存在 → 从 JSON 加载，用 np.array() 把列表转回 numpy 数组
    if os.path.exists(BANDIT_PATH):
        with open(BANDIT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _bandit = LinUCB(data["n_actions"], data["n_features"], data["alpha"])
        _bandit.total_updates = data["total_updates"]
        _bandit.A = [np.array(a) for a in data["A"]]
        _bandit.b = [np.array(b) for b in data["b"]]
    else:
        # 如果文件不存在 → 新建一个空模型
        _bandit = LinUCB()
    return _bandit

def _save_bandit():
    bandit = _get_bandit()
    data = {
        "n_actions": bandit.n_actions,
        "n_features": bandit.n_features,
        "alpha": bandit.alpha,
        "total_updates": bandit.total_updates,
        "A": [a.tolist() for a in bandit.A],
        "b": [b.tolist() for b in bandit.b],
    }
    # 把当前模型的所有参数存成 JSON。
    with open(BANDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_query_features(query, complexity):
    complexity_map = {"simple": 0.0, "medium": 0.5, "complex": 1.0}
    complexity_score = complexity_map.get(complexity, 0.5)

    local_keywords = ["Docker", "docker", "容器", "Python", "python",
                      "C++", "c++", "C#", "c#", "编程", "镜像", "Dockerfile"]
    local_relevance = 1.0 if any(kw in query for kw in local_keywords) else 0.0

    length_norm = min(len(query) / 100, 1.0)

    return np.array([complexity_score, local_relevance, length_norm])

ACTIONS = ["local", "web", "both"]

def bandit_select_routing(query, complexity):
    bandit = _get_bandit()
    features = extract_query_features(query, complexity)
    action = bandit.select_action(features)
    return ACTIONS[action]

def bandit_update_reward(query, complexity, routing, reward):
    bandit = _get_bandit()
    features = extract_query_features(query, complexity)
    action = ACTIONS.index(routing)
    bandit.update(action, features, reward)
    _save_bandit()