import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

sessions = []
with open("sessions.jsonl") as f:
    for line in f:
        line = line.strip()
        if line:
            sessions.append(json.loads(line))

# основные показатели
def indicators(sessions):
    # количество сессий 
    all_sessions = len(sessions)

    # информация про уникальные и повторяющиеся товары
    all_items = []
    session_lengh = []
    repeated_sessions = 0

    for session in sessions:
        all_items.extend(session)
        session_lengh.append(len(session))
        if len(session) != len(set(session)):
            repeated_sessions += 1

    unique_items = set(all_items)
    repeated_items = repeated_sessions / len(sessions) * 100

    # среднее, медиана, макс и мин
    mean_lengh = np.mean(session_lengh)
    median_lengh = np.median(session_lengh)
    mx_lengh = max(session_lengh)
    mn_lengh = min(session_lengh)

    # оценка популярности по уникальным просмотрам
    unique_views = Counter()

    for session in sessions:
        unique_session = set(session)
        unique_views.update(unique_session)

    mn_freq = min(unique_views.values())
    mx_freq = max(unique_views.values())
    mn_freq_items = []
    mx_freq_items = []

    for item, freq in unique_views.items():
        if freq == mn_freq:
            mn_freq_items.append(item)
        if freq == mx_freq:
            mx_freq_items.append(item)

    top_10 = [item[0] for item in unique_views.most_common(10)]

    return all_sessions, unique_items, repeated_items, unique_views, mean_lengh, median_lengh, mx_lengh, mn_lengh, mx_freq_items, mn_freq_items, top_10

all_sessions, unique_items, repeated_items, unique_views, mean_lengh, median_lengh, mx_lengh, mn_lengh, mx_freq_items, mn_freq_items, top_10 = indicators(sessions)

#графики
def graphs(sessions, unique_views):
    items = list(unique_views.keys())
    unique_views_count = list(unique_views.values())
    plt.bar(items, unique_views_count, color="green", alpha=0.7)
    plt.xlabel("item ID")
    plt.ylabel("views")
    plt.title("the number of views per item")
    plt.show()

    lengths = [len(session) for session in sessions]
    plt.hist(lengths, bins=50, color="blue", alpha=0.7)
    plt.xlabel("session`s length")
    plt.ylabel("session`s count")
    plt.title("session length distribution")
    plt.show()

# train/test
def train_test_split(
    sessions: list[list[int]],
) -> tuple[list[list[int]], list[int]]:
    train_sessions = [session[:-1] for session in sessions]
    test_targets = [session[-1] for session in sessions]

    return train_sessions, test_targets

train_sessions, test_targets = train_test_split(sessions)

# граф переходов
def transition_graph(train_sessions):
    ver = Counter() #сколько у вершины ребер всего
    edg = Counter() #сколько у curr и next ребер всего
    all_items = set()

    for session in train_sessions:
        for i in range(len(session) - 1):
            curr = session[i]
            next = session[i + 1]
            ver[curr] += 1
            edg[(curr, next)] += 1
            all_items.add(curr)
            all_items.add(next)
    
    adj_list = {item: {} for item in all_items} #список смежности из вершины curr и вероятности перехода к next

    for (curr, next), count_edges in edg.items():
        adj_list[curr][next] = count_edges / ver[curr]
    
    # обратными ребрами заполняем не встретившиеся переходы
    for item in all_items:
        if not adj_list[item]:
            inverse_edg = {} 
            for (curr, next), count_edges in edg.items():
                if next == item:
                    inverse_edg[curr] = count_edges / ver[curr]
            adj_list[item] = inverse_edg

    return adj_list 

# рекомендательная модель для топа 10
def recommendation_model_top10(train_sessions, adj_list):
    all_recommendations = []
    for session in train_sessions:
        last_item = session[-1]
        edg_prob = adj_list[last_item]
        sort_edg_prob = sorted(edg_prob.items(), key= lambda x: x[1], reverse=True)
        top_10 = sort_edg_prob[:10]
        recommendations_for_session = [item[0] for item in top_10]
        all_recommendations.append(recommendations_for_session)
    
    return all_recommendations

# hit at 
def hit_at_k(
    recommendations: list[list[int]],
    true_items: list[int],
    k: int = 10,
) -> float:
    
    assert len(recommendations) == len(true_items)

    hits = 0
    for recs, true_item in zip(recommendations, true_items):
        if true_item in recs[:k]:
            hits += 1

    return hits / len(true_items)  

def baseline(train_sessions, test_targets):
    views = Counter()
    for session in train_sessions:
        views.update(session)
    
    top = [item[0] for item in views.most_common(10)]
    baseline_rec = [top for target in test_targets]
    return baseline_rec


# вызов параметров
print(f"Всего сессий: {all_sessions}")
print(f"Всего уникальных товаров: {len(unique_items)}")
print(f"Процент повторяющихся товаров в сессиях: {repeated_items:.2f}")
print(f"Средняя длина сессии: {mean_lengh:.2f}")
print(f"Медиана длин сессий: {median_lengh:.2f}")
print(f"Максимальная длина сессии: {mx_lengh}")
print(f"Минимальная длина сессии: {mn_lengh}")
print("Самый непопулярный товар:", ", ".join(str(x) for x in mn_freq_items))
print("Самый популярный товар:", ", ".join(str(x) for x in mx_freq_items))
print("Топ 10 самых популярных товаров:", ", ".join(str(x) for x in top_10))

adj_list = transition_graph(train_sessions)
recommendations = recommendation_model_top10(train_sessions, adj_list)
baseline_rec = baseline(train_sessions, test_targets)
hit_b = hit_at_k(baseline_rec, test_targets)
hit_m = hit_at_k(recommendations, test_targets)
print(f"Оценка качества baseline: {hit_b}")
print(f"Оценка качества модели: {hit_m}")

graphs(sessions, unique_views)