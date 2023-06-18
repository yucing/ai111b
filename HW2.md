# 此程式碼為參考魏仲彥同學，看完並理解
```py
import math
import random

input = {1: [0, 0],
         2: [1, 1],
         3: [2, 2],
         4: [3, 3],
         5: [1, 0],
         6: [0, 1],
         7: [1, 2],
         8: [2, 1],
         9: [1, 3],
         10: [3, 1],
         11: [2, 3],
         12: [3, 2], }


class Graph:
    """
    存圖像的class
    輸入的點，使用字典做儲存，key為點，value為[x, y]
    :param graph: 輸入圖形
    :return graph: 輸出圖形
    """
    def __init__(self, graph):
        self.g = graph

    def getGraph(self):
        return self.g


class TSP:
    """
    搜尋最短路徑，主要觸發函數是calculate
    :param graph: 輸入圖形
    :return : 輸出圖形
    """
    def __init__(self, graph):
        self.graph = graph  # 字典，存圖
        self.score = 0  # 存路徑長度。越低越好
        self.order = []  # 存順序，就是存取上面字典的key
    def calculate(self, tolerate, times):
        """
        搜尋最短路徑，主要觸發函數是calculate
        :param tolerate: 比較次數，如果跑多次都大於原本的值，就退出並return最好的結果
        :param times: swap的次數 
        :return [score, order]: 回傳路徑長度(越低越好)和路徑排序
        """
        self.ListInit()
        l = len(self.order)
        pre_score = self.score
        pre_order = self.order
        tol = tolerate
        while True:
            self.score = 0
            # self.order = self.swap()
            for i in range(l - 1):
                now_pos = self.graph[self.order[i]]
                next_pos = self.graph[self.order[i + 1]]
                h = self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
                for _ in range(times):
                    # 使用random做swap
                    new_order = self.swap(i+1, random.randint(i+1, l-1))
                    now_pos = self.graph[new_order[i]]
                    next_pos = self.graph[new_order[i + 1]]
                    nh = self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
                    if h > nh:
                        self.order = new_order
                self.score += self.distence(now_pos[0] - next_pos[0], now_pos[1] - next_pos[1])
            if not tol:
                return [pre_score, pre_order]

            if self.score < pre_score:
                if not tol:
                    return [self.score, self.order]
                else:
                    # print(tol)
                    pre_order = self.order
                    pre_score = self.score
                    tol = tolerate
            elif not pre_score:
                pre_score = self.score
            else:
                tol -= 1


    def distence(self, x, y):
        """
        計算距離
        :param x: 兩點相減後的x
        :param y: 兩點相減後的y
        :return distence: 兩點的路徑
        """
        return math.sqrt(x ** 2 + y ** 2)

    def swap(self, a, b):
        """
        兩點做交換，上方的calculate裡面是使用random與除第一點外的點做交換
        :param a: 交換成b
        :param b: 交換成a
        :return new_order: 新的順序
        """
        l = len(self.order)
        new_order = []
        for i in range(l):
            if a == i:
                new_order.append(self.order[b])
            elif b == i:
                new_order.append(self.order[a])
            else:
                new_order.append(self.order[i])

        return new_order

    def ListInit(self):
        """
        初始化順序(self.order)
        :return : 初始順序
        """
        for k, v in self.graph.items():
            self.order.append(k)


graph = Graph(input)
ans = TSP(graph.getGraph())
print(graph.getGraph())
print(ans.calculate(10, 100))  # 裡面的tolerate和times越大，就會算越準
```