# 狼、羊、菜過河問題
## 程式碼由Chat GPT產生，並進行修改
```py
objs = ["人", "狼", "羊", "菜"]
state = [0, 0, 0, 0]

def neighbors(s):
    side = s[0]
    next_states = []
    check_add(next_states, move(s, 0))
    for i in range(1, len(s)):
        if s[i] == side:
            check_add(next_states, move(s, i))
    return next_states

def check_add(next_states, s):
    if not is_dead(s):
        next_states.append(s)

def is_dead(s):
    if s[1] == s[2] and s[1] != s[0]:
        return True  # 狼吃羊
    if s[2] == s[3] and s[2] != s[0]:
        return True  # 羊吃菜
    return False

# 人帶著 obj 移到另一邊
def move(s, obj):
    new_s = s.copy()  # 複製一份陣列
    side = s[0]
    another_side = 1 if side == 0 else 0
    new_s[0] = another_side
    new_s[obj] = another_side
    return new_s

visited_map = {}

def visited(s):
    s_str = ''.join(map(str, s))
    return s_str in visited_map

def is_success(s):
    for i in range(len(s)):
        if s[i] == 0:
            return False
    return True

def state2str(s):
    st = ""
    for i in range(len(s)):
        st += objs[i] + str(s[i]) + " "
    return st

path = []

def print_path(path):
    for p in path:
        print(state2str(p))

def dfs(s):
    if visited(s):
        return
    path.append(s)
    if is_success(s):
        print("success!")
        print_path(path)
        return
    visited_map[''.join(map(str, s))] = True
    neighbors_list = neighbors(s)
    for neighbor in neighbors_list:
        dfs(neighbor)
    path.pop()

dfs(state)
```

# 輸出結果

![](https://github.com/yucing/ai111b/blob/main/picture/2.jpg)